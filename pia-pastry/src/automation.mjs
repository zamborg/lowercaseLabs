import fs from "node:fs";
import path from "node:path";
import { spawn } from "node:child_process";
import { createInterface } from "node:readline/promises";
import { stdin as input, stdout as output } from "node:process";
import { chromium } from "playwright";
import { AUTO_PURCHASE_ACK } from "./config.mjs";
import {
  chooseNextDropTitle,
  choosePreferredLabel,
  chooseRandomMatchingLabel,
  extractDisplayedTotal,
  isFinalPurchaseLabel,
  isPickupDateLabel,
  isPickupTimeLabel,
  pickProduct,
} from "./logic.mjs";

const sleep = (milliseconds) =>
  new Promise((resolve) => setTimeout(resolve, milliseconds));

function timestamp() {
  return new Date().toISOString();
}

function safeFileTimestamp() {
  return timestamp().replaceAll(":", "-").replaceAll(".", "-");
}

function log(event, details = {}) {
  const record = { at: timestamp(), event, ...details };
  console.log(JSON.stringify(record));
  return record;
}

function appendRecord(run, record) {
  fs.appendFileSync(run.logFile, `${JSON.stringify(record)}\n`);
}

async function record(run, event, details = {}, page = null) {
  const entry = log(event, details);
  appendRecord(run, entry);
  if (page && ["drop-found", "handoff", "failure", "purchased"].includes(event)) {
    const screenshot = path.join(
      run.directory,
      `${safeFileTimestamp()}-${event}.png`,
    );
    await page.screenshot({ path: screenshot, fullPage: true }).catch(() => {});
  }
  return entry;
}

function notify(title, message) {
  process.stdout.write("\u0007");
  log("notification", { title, message });

  if (process.platform !== "darwin") return;
  const clean = (value) => value.replaceAll("\\", "\\\\").replaceAll('"', '\\"');
  const script = `display notification "${clean(message)}" with title "${clean(title)}"`;
  spawn("osascript", ["-e", script], {
    detached: true,
    stdio: "ignore",
  }).unref();
}

function createRun(config) {
  const directory = path.join(config.artifactDir, safeFileTimestamp());
  fs.mkdirSync(directory, { recursive: true });
  return {
    directory,
    logFile: path.join(directory, "run.jsonl"),
  };
}

async function openBrowser(config) {
  fs.mkdirSync(config.profileDir, { recursive: true });
  const launchOptions = {
    headless: config.headless,
    viewport: null,
  };
  if (config.browserChannel !== "chromium") {
    launchOptions.channel = config.browserChannel;
  }

  const context = await chromium.launchPersistentContext(
    config.profileDir,
    launchOptions,
  );
  const pages = context.pages();
  const page = pages[0] ?? (await context.newPage());
  page.setDefaultTimeout(3_000);
  page.setDefaultNavigationTimeout(12_000);
  return { context, page };
}

async function prompt(message) {
  const terminal = createInterface({ input, output });
  try {
    return await terminal.question(message);
  } finally {
    terminal.close();
  }
}

async function waitUntilArmed(config, run) {
  if (!config.releaseAt) return;
  const wakeAt =
    config.releaseAt.valueOf() - Math.max(config.leadSeconds, 0) * 1_000;

  while (Date.now() < wakeAt) {
    const remainingMs = wakeAt - Date.now();
    await record(run, "sleeping-until-release", {
      wakeAt: new Date(wakeAt).toISOString(),
      remainingSeconds: Math.ceil(remainingMs / 1_000),
    });
    await sleep(Math.min(remainingMs, 30_000));
  }
}

async function controlLabel(control) {
  const innerText = (await control.innerText().catch(() => "")).trim();
  if (innerText) return innerText;
  return control.evaluate((element) => {
    const ariaLabel = element.getAttribute("aria-label");
    if (ariaLabel) return ariaLabel;

    const labelledBy = element.getAttribute("aria-labelledby");
    if (labelledBy) {
      const text = labelledBy
        .split(/\s+/u)
        .map((id) => document.getElementById(id)?.textContent ?? "")
        .join(" ")
        .trim();
      if (text) return text;
    }

    if ("labels" in element && element.labels) {
      const text = [...element.labels]
        .map((label) => label.textContent ?? "")
        .join(" ")
        .trim();
      if (text) return text;
    }
    return element.getAttribute("title") ?? "";
  });
}

async function visibleEnabledButtonLabels(root) {
  const buttons = root.getByRole("button");
  const count = Math.min(await buttons.count(), 80);
  const labels = [];
  for (let index = 0; index < count; index += 1) {
    const button = buttons.nth(index);
    if ((await button.isVisible()) && (await button.isEnabled())) {
      labels.push((await controlLabel(button)).replace(/\s+/gu, " ").trim());
    }
  }
  return labels.filter(Boolean);
}

async function clickUniqueButtonContaining(root, text) {
  const locator = root
    .getByRole("button")
    .filter({ hasText: text })
    .filter({ visible: true });
  const count = await locator.count();
  if (count !== 1) return false;
  if (!(await locator.isEnabled())) return false;
  await locator.click();
  return true;
}

async function pageText(page) {
  return page.locator("body").innerText({ timeout: 3_000 });
}

async function waitForHotplateRender(page) {
  await page
    .locator("h1")
    .waitFor({ state: "visible", timeout: 8_000 })
    .catch(() => {});
}

async function hasHumanChallenge(page) {
  const challengeFrames = page.locator(
    'iframe[src*="captcha"], iframe[src*="recaptcha"], iframe[src*="hcaptcha"], iframe[src*="turnstile"]',
  );
  const frameCount = Math.min(await challengeFrames.count(), 10);
  for (let index = 0; index < frameCount; index += 1) {
    if (await challengeFrames.nth(index).isVisible()) return true;
  }
  const text = await pageText(page);
  return /captcha|verify (?:that )?you are human|security check/iu.test(text);
}

async function hasManualAuthStep(page) {
  const text = await pageText(page);
  if (/verification code|one[- ]time code|enter.*code|card number/iu.test(text)) {
    return true;
  }

  const unfilledSensitiveInputs = page.locator(
    [
      'input[autocomplete="one-time-code"]',
      'input[autocomplete="tel"]',
      'input[autocomplete="cc-number"]',
      'input[name*="otp" i]',
      'input[name*="card" i]',
      'input[placeholder*="phone" i]',
      'input[placeholder*="verification" i]',
    ].join(","),
  );
  const count = Math.min(await unfilledSensitiveInputs.count(), 20);
  for (let index = 0; index < count; index += 1) {
    const field = unfilledSensitiveInputs.nth(index);
    if (!(await field.isVisible())) continue;
    if ((await field.inputValue().catch(() => "")).trim() === "") return true;
  }
  return false;
}

async function humanHandoff(config, run, page, reason) {
  notify("pia-pastry needs you", reason);
  await record(run, "handoff", { reason, url: page.url() }, page);
  if (config.headless) {
    throw new Error(
      `${reason} Re-run with PIA_HEADLESS=false so the step can be completed.`,
    );
  }
  await prompt(`${reason}\nComplete it in the browser, then press Enter here… `);
}

async function choosePreferenceControl(
  root,
  preferences,
  role,
  alreadyChosen,
  predicate = () => true,
) {
  if (preferences.length === 0) return false;
  const controls = root.getByRole(role);
  const count = Math.min(await controls.count(), 60);
  const candidates = [];

  for (let index = 0; index < count; index += 1) {
    const control = controls.nth(index);
    if (!(await control.isVisible()) || !(await control.isEnabled())) continue;
    const accessibleLabel = (await controlLabel(control)).trim();
    if (
      !accessibleLabel ||
      alreadyChosen.has(accessibleLabel) ||
      !predicate(accessibleLabel)
    ) {
      continue;
    }

    const selected =
      (role === "radio" && (await control.isChecked().catch(() => false))) ||
      (await control.getAttribute("aria-pressed")) === "true" ||
      (await control.getAttribute("aria-selected")) === "true" ||
      (await control.getAttribute("data-state")) === "checked" ||
      (await control.getAttribute("data-selected")) === "true";
    if (selected) {
      alreadyChosen.add(accessibleLabel);
      continue;
    }

    candidates.push({ index, label: accessibleLabel });
  }

  const selected = choosePreferredLabel(
    candidates.map(({ label }) => label),
    preferences,
  );
  if (!selected) return false;
  const match = candidates.find(({ label }) => label === selected);
  await controls.nth(match.index).click();
  alreadyChosen.add(selected);
  return true;
}

async function chooseRandomControl(
  root,
  role,
  predicate,
  alreadyChosen,
) {
  const controls = root.getByRole(role);
  const count = Math.min(await controls.count(), 60);
  const candidates = [];

  for (let index = 0; index < count; index += 1) {
    const control = controls.nth(index);
    if (!(await control.isVisible()) || !(await control.isEnabled())) continue;
    const label = (await controlLabel(control)).replace(/\s+/gu, " ").trim();
    if (!label || alreadyChosen.has(label) || !predicate(label)) continue;

    const selected =
      (role === "radio" && (await control.isChecked().catch(() => false))) ||
      (await control.getAttribute("aria-pressed")) === "true" ||
      (await control.getAttribute("aria-selected")) === "true" ||
      (await control.getAttribute("data-state")) === "checked" ||
      (await control.getAttribute("data-selected")) === "true";
    if (!selected) candidates.push({ index, label });
  }

  const selected = chooseRandomMatchingLabel(
    candidates.map(({ label }) => label),
    () => true,
  );
  if (!selected) return null;
  const match = candidates.find(({ label }) => label === selected);
  await controls.nth(match.index).click();
  alreadyChosen.add(selected);
  return selected;
}

async function setQuantity(page, quantity) {
  if (quantity <= 1) return;
  const buttons = await visibleEnabledButtonLabels(page);
  const plusLabel = buttons.find(
    (label) => /^\+$/u.test(label) || /\bincrease\b|\badd one\b/iu.test(label),
  );
  if (!plusLabel) {
    throw new Error("Quantity control was not found.");
  }
  for (let count = 1; count < quantity; count += 1) {
    if (!(await clickUniqueButtonContaining(page, plusLabel))) {
      throw new Error("Quantity control became ambiguous.");
    }
  }
}

function addToCartTimeout(releaseAt) {
  if (!releaseAt) return 3_000;
  return Math.max(releaseAt.valueOf() + 2_000 - Date.now(), 3_000);
}

async function waitForAddToCart(dialog, releaseAt) {
  const button = dialog.getByRole("button", { name: /^Add to Cart\b/iu });
  await button.waitFor({
    state: "visible",
    timeout: addToCartTimeout(releaseAt),
  });
  return button.isEnabled();
}

async function slamAddToCart(config, run, dialog, product) {
  const button = dialog.getByRole("button", { name: /^Add to Cart\b/iu });
  try {
    await button.click({ timeout: addToCartTimeout(config.releaseAt) });
    await record(run, "cart-reserved", { product });
    await dialog.waitFor({ state: "hidden", timeout: 2_000 }).catch(() => {});
    return true;
  } catch {
    return false;
  }
}

async function enterNextDrop(page, run) {
  await page
    .getByRole("heading", { name: "Next Drop", exact: true })
    .waitFor({ state: "visible", timeout: 5_000 })
    .catch(() => {});
  const headings = await page.locator("h2").allTextContents();
  const title = chooseNextDropTitle(headings);
  if (!title) return false;

  const allHeadings = page.locator("h2");
  const count = await allHeadings.count();
  const headingIndex = headings.findIndex((text) => text.trim() === title);
  if (headingIndex < 0 || headingIndex >= count) return false;

  await record(run, "drop-found", { title }, page);
  const heading = allHeadings.nth(headingIndex);
  await Promise.all([
    page.waitForURL((url) => url.pathname.split("/").filter(Boolean).length >= 2),
    heading.click(),
  ]);
  return true;
}

async function selectProduct(config, run, page) {
  await page
    .getByRole("heading", { name: config.products[0], exact: true })
    .waitFor({ state: "visible", timeout: 5_000 })
    .catch(() => {});
  const labels = await visibleEnabledButtonLabels(page);
  const availableLabels = labels.filter((label) =>
    config.products.some((product) =>
      label.toLocaleLowerCase().includes(product.toLocaleLowerCase()),
    ),
  );

  for (const product of config.products) {
    const selection = pickProduct(availableLabels, [product]);
    if (!selection) continue;

    await record(run, "product-found", selection);
    if (!(await clickUniqueButtonContaining(page, product))) continue;
    const dialog = page.getByRole("dialog");
    await dialog.waitFor({ state: "visible", timeout: 3_000 }).catch(() => {});
    const currentText = await pageText(page);

    if (
      /sales have not begun|item is not available yet|available soon/iu.test(
        currentText,
      )
    ) {
      await record(run, "product-not-open", { product });
      await setQuantity(dialog, config.quantity);
      if (config.mode === "dry-run") {
        if (await waitForAddToCart(dialog, config.releaseAt).catch(() => false)) {
          return { status: "observed-live-product", product };
        }
      } else if (await slamAddToCart(config, run, dialog, product)) {
        return { status: "cart-reserved", product };
      }
      await page.keyboard.press("Escape").catch(() => {});
      return { status: "not-open", product };
    }

    if (/all in (?:other )?carts|currently in (?:other )?carts/iu.test(currentText)) {
      await record(run, "product-in-other-carts", { product });
      await page.keyboard.press("Escape").catch(() => {});
      return { status: "in-other-carts", product };
    }

    if (/this item is currently sold out|\bsold out\b/iu.test(currentText)) {
      await record(run, "product-sold-out", { product });
      if (config.mode === "dry-run") {
        return { status: "rehearsed-sold-out", product };
      }
      await page.keyboard.press("Escape").catch(() => {});
      continue;
    }

    if (config.mode === "dry-run") {
      return { status: "observed-live-product", product };
    }

    await setQuantity(dialog, config.quantity);
    if (await slamAddToCart(config, run, dialog, product)) {
      return { status: "cart-reserved", product };
    }
    await page.keyboard.press("Escape").catch(() => {});
    return { status: "cart-race-lost", product };
  }

  return { status: "no-product" };
}

async function clickActionButton(root, patterns) {
  const labels = await visibleEnabledButtonLabels(root);
  const label = labels.find(
    (candidate) =>
      patterns.some((pattern) => pattern.test(candidate)) &&
      !isFinalPurchaseLabel(candidate),
  );
  if (!label) return null;
  if (!(await clickUniqueButtonContaining(root, label))) return null;
  return label;
}

async function findFinalPurchaseButton(root) {
  const labels = await visibleEnabledButtonLabels(root);
  const label = labels.find(isFinalPurchaseLabel);
  if (!label) return null;
  const locator = root
    .getByRole("button")
    .filter({ hasText: label })
    .filter({ visible: true });
  if ((await locator.count()) !== 1) return null;
  return { label, locator };
}

async function handleFinalPurchase(config, run, page, button) {
  const text = await pageText(page);
  const total = extractDisplayedTotal(text);

  if (config.mode !== "auto") {
    await humanHandoff(
      config,
      run,
      page,
      `Order is ready for review${total == null ? "" : ` at $${total.toFixed(2)}`}.`,
    );
    return { status: "ready-for-review", total };
  }

  if (config.autoPurchaseAcknowledgement !== AUTO_PURCHASE_ACK) {
    await humanHandoff(
      config,
      run,
      page,
      `Auto mode is not armed. Set PIA_AUTO_PURCHASE_ACK=${AUTO_PURCHASE_ACK} or place the order manually.`,
    );
    return { status: "auto-not-armed", total };
  }

  if (total == null) {
    await humanHandoff(
      config,
      run,
      page,
      "The displayed order total could not be verified; place the order manually.",
    );
    return { status: "total-unverified" };
  }

  if (total > config.maxTotalDollars) {
    await humanHandoff(
      config,
      run,
      page,
      `The $${total.toFixed(2)} total exceeds the $${config.maxTotalDollars.toFixed(2)} cap.`,
    );
    return { status: "over-cap", total };
  }

  await record(run, "placing-order", { total, label: button.label });
  await button.locator.click();
  await page.waitForLoadState("domcontentloaded").catch(() => {});
  const confirmation = await pageText(page);
  if (!/order confirmed|thank you|order number|pickup confirmed/iu.test(confirmation)) {
    await humanHandoff(
      config,
      run,
      page,
      "The final click happened, but confirmation was not detected. Check the browser now.",
    );
    return { status: "confirmation-unclear", total };
  }

  notify("pia-pastry", "Butter & Crumble order confirmed.");
  await record(run, "purchased", { total, url: page.url() }, page);
  return { status: "purchased", total };
}

async function advanceCheckout(config, run, page) {
  const chosenDayLabels = new Set();
  const chosenTimeLabels = new Set();
  for (let step = 0; step < 14; step += 1) {
    if (await hasHumanChallenge(page)) {
      await humanHandoff(
        config,
        run,
        page,
        "Hotplate presented a human-verification step.",
      );
      continue;
    }
    if (await hasManualAuthStep(page)) {
      await humanHandoff(
        config,
        run,
        page,
        "Hotplate needs account, verification, or payment input.",
      );
      continue;
    }

    const dialog = page.getByRole("dialog");
    const scope =
      (await dialog.count()) === 1 && (await dialog.isVisible()) ? dialog : page;

    const finalButton = await findFinalPurchaseButton(scope);
    if (finalButton) {
      return handleFinalPurchase(config, run, page, finalButton);
    }

    const choseDay =
      (await choosePreferenceControl(
        scope,
        config.pickupPreferences,
        "radio",
        chosenDayLabels,
        isPickupDateLabel,
      )) ||
      (await choosePreferenceControl(
        scope,
        config.pickupPreferences,
        "button",
        chosenDayLabels,
        isPickupDateLabel,
      ));
    if (choseDay) {
      await record(run, "pickup-day-selected");
      continue;
    }

    if (config.randomPickup) {
      let randomDay = null;
      for (const role of ["radio", "button", "option"]) {
        randomDay = await chooseRandomControl(
          scope,
          role,
          isPickupDateLabel,
          chosenDayLabels,
        );
        if (randomDay) break;
      }
      if (randomDay) {
        await record(run, "pickup-day-selected", {
          strategy: "random",
          label: randomDay,
        });
        continue;
      }
    }

    const choseTime =
      (await choosePreferenceControl(
        scope,
        config.pickupTimePreferences,
        "radio",
        chosenTimeLabels,
        isPickupTimeLabel,
      )) ||
      (await choosePreferenceControl(
        scope,
        config.pickupTimePreferences,
        "button",
        chosenTimeLabels,
        isPickupTimeLabel,
      ));
    if (choseTime) {
      await record(run, "pickup-time-selected");
      continue;
    }

    if (config.randomPickup) {
      let randomTime = null;
      for (const role of ["radio", "button", "option"]) {
        randomTime = await chooseRandomControl(
          scope,
          role,
          isPickupTimeLabel,
          chosenTimeLabels,
        );
        if (randomTime) break;
      }
      if (randomTime) {
        await record(run, "pickup-time-selected", {
          strategy: "random",
          label: randomTime,
        });
        continue;
      }
    }

    const action = await clickActionButton(scope, [
      /^add(?:\s+to\s+(?:bag|cart|order))?\b/iu,
      /^(?:view\s+)?(?:bag|cart|order)\b/iu,
      /^checkout\b/iu,
      /^(?:continue|next|review order)\b/iu,
    ]);
    if (action) {
      await record(run, "checkout-action", { action });
      await page.waitForTimeout(150);
      continue;
    }

    await humanHandoff(
      config,
      run,
      page,
      "The next safe checkout control was not recognized.",
    );
  }

  throw new Error("Checkout exceeded the guarded step limit.");
}

async function pollDelay(config, attempt) {
  const backoff = attempt > 40 ? Math.min((attempt - 40) * 150, 2_500) : 0;
  const jitter = Math.floor(Math.random() * Math.max(config.pollJitterMs, 0));
  await sleep(config.pollMs + backoff + jitter);
}

export async function runSetup(config) {
  const { context, page } = await openBrowser({ ...config, headless: false });
  try {
    await page.goto(config.storeUrl, { waitUntil: "domcontentloaded" });
    notify("pia-pastry setup", "Prepare the dedicated Hotplate browser profile.");
    console.log(
      [
        "",
        "In the browser:",
        "  1. Sign in to Hotplate if the site offers it.",
        "  2. Complete any phone verification.",
        "  3. Let the browser save only the checkout details you want persisted.",
        "  4. Return here and press Enter.",
        "",
        "No phone number, verification code, or card data is written to .env.",
      ].join("\n"),
    );
    await prompt("Press Enter when the dedicated profile is ready… ");
  } finally {
    await context.close();
  }
}

export async function runDrop(config) {
  const run = createRun(config);
  const { context, page } = await openBrowser(config);
  await record(run, "started", {
    command: config.command,
    mode: config.mode,
    storeUrl: config.storeUrl,
    eventUrl: config.eventUrl,
    products: config.products,
  });

  try {
    await waitUntilArmed(config, run);
    await page.goto(config.eventUrl ?? config.storeUrl, {
      waitUntil: "domcontentloaded",
    });
    await waitForHotplateRender(page);

    for (let attempt = 0; ; attempt += 1) {
      const url = new URL(page.url());
      const storePath = new URL(config.storeUrl).pathname.replace(/\/$/u, "");
      const onStore = url.pathname.replace(/\/$/u, "") === storePath;

      if (onStore) {
        if (await enterNextDrop(page, run)) continue;
        if (config.command === "rehearse") {
          throw new Error("Rehearsal event did not open.");
        }
        if (attempt % 10 === 0) {
          await record(run, "waiting-for-drop", { attempt, url: page.url() });
        }
        await pollDelay(config, attempt);
        await page.reload({ waitUntil: "domcontentloaded" });
        await waitForHotplateRender(page);
        continue;
      }

      const text = await pageText(page);
      if (/sales have ended for this drop/iu.test(text) && config.command !== "rehearse") {
        await record(run, "stale-drop", { url: page.url() });
        await page.goto(config.storeUrl, { waitUntil: "domcontentloaded" });
        await waitForHotplateRender(page);
        continue;
      }

      const product = await selectProduct(config, run, page);
      if (
        product.status === "rehearsed-sold-out" ||
        product.status === "observed-live-product"
      ) {
        await record(run, "rehearsal-complete", product, page);
        return product;
      }
      if (product.status === "cart-reserved") {
        return await advanceCheckout(config, run, page);
      }

      if (attempt % 10 === 0) {
        await record(run, "waiting-for-product", {
          attempt,
          status: product.status,
          url: page.url(),
        });
      }
      await pollDelay(config, attempt);
      await page.reload({ waitUntil: "domcontentloaded" });
      await waitForHotplateRender(page);
    }
  } catch (error) {
    await record(
      run,
      "failure",
      {
        message: error instanceof Error ? error.message : String(error),
        url: page.url(),
      },
      page,
    );
    throw error;
  } finally {
    await context.close();
  }
}
