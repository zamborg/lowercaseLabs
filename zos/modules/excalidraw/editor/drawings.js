const list = document.querySelector("main");
const message = document.querySelector("#status");
document.addEventListener("submit", async event => {
  event.preventDefault();
  const form = event.target;
  const action = event.submitter.value;
  if (action === "delete" && !confirm("Delete this drawing?")) return;
  const body = {action, title:form.elements.title.value};
  if (form.dataset.version) body.if_version = Number(form.dataset.version);
  try {
    const res = await fetch(form.action, {method:"POST", headers:{"Content-Type":"application/json",
      "X-CSRF-Token":list.dataset.csrf}, body:JSON.stringify(body)});
    const data = await res.json();
    if (!res.ok) throw new Error(res.status === 409 ? "Drawing changed. Reload before trying again." : data.error);
    if (action === "create") location.assign(list.dataset.prefix + "/d/" + data.id);
    else location.reload();
  } catch (err) { message.textContent = err.message; }
});
