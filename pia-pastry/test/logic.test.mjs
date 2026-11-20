import test from "node:test";
import assert from "node:assert/strict";
import {
  chooseNextDropTitle,
  choosePreferredLabel,
  chooseRandomMatchingLabel,
  extractDisplayedTotal,
  isFinalPurchaseLabel,
  isPickupDateLabel,
  isPickupTimeLabel,
  pickProduct,
} from "../src/logic.mjs";

test("chooses only the card inside Next Drop", () => {
  assert.equal(
    chooseNextDropTitle([
      "Next Drop",
      "Pastry Preorders Thurs - Sun, 8/6 - 8/9!",
      "Frequently Asked Questions",
      "Past drop that must be ignored",
    ]),
    "Pastry Preorders Thurs - Sun, 8/6 - 8/9!",
  );
});

test("returns null when the next drop is not scheduled", () => {
  assert.equal(
    chooseNextDropTitle([
      "Next Drop",
      "Butter And Crumble hasn't scheduled their next drop yet. Signup to be reminded when they do!",
      "Frequently Asked Questions",
    ]),
    null,
  );
});

test("selects products in configured preference order", () => {
  assert.deepEqual(
    pickProduct(
      [
        "Large Pastry Box $68.00",
        "Small Pastry Box $32.00 Sold Out",
      ],
      ["Small Pastry Box", "Large Pastry Box"],
    ),
    {
      product: "Small Pastry Box",
      label: "Small Pastry Box $32.00 Sold Out",
      soldOut: true,
    },
  );
});

test("chooses pickup labels by preference, not DOM order", () => {
  assert.equal(
    choosePreferredLabel(
      ["Saturday, 10:00 AM", "Thursday, 2:00 PM"],
      ["Thursday", "Saturday"],
    ),
    "Thursday, 2:00 PM",
  );
});

test("extracts the explicit total rather than item prices", () => {
  assert.equal(
    extractDisplayedTotal(
      "Small box $32.00\nService fee $2.40\nOrder total $37.89",
    ),
    37.89,
  );
  assert.equal(extractDisplayedTotal("Small box $32.00"), null);
});

test("recognizes consequential final buttons", () => {
  assert.equal(isFinalPurchaseLabel("Place order · $37.89"), true);
  assert.equal(isFinalPurchaseLabel("Review order"), false);
});

test("recognizes compact pickup dates without matching product copy", () => {
  assert.equal(isPickupDateLabel("Wednesday, August 12"), true);
  assert.equal(isPickupDateLabel("Tomorrow"), true);
  assert.equal(
    isPickupDateLabel(
      "Small Pastry Boxes, Wednesday Preorders with a very long description that is not a slot",
    ),
    false,
  );
});

test("recognizes pickup times", () => {
  assert.equal(isPickupTimeLabel("9:30 AM"), true);
  assert.equal(isPickupTimeLabel("10:00 AM · 4 spots left"), true);
  assert.equal(isPickupTimeLabel("Add to Cart $32.00"), false);
});

test("random matching label is bounded to matching candidates", () => {
  assert.equal(
    chooseRandomMatchingLabel(
      ["Checkout", "9:00 AM", "9:30 AM"],
      isPickupTimeLabel,
      () => 0.99,
    ),
    "9:30 AM",
  );
});
