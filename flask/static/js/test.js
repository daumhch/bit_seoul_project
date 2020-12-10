const keys = document.querySelectorAll(".key"),
    note = document.querySelector(".nowplaying")

function playNote(e) {
    key = document.querySelector(`.key[data-key="${e.keyCode}"]`);
    if (!key) return;

    const keyNote = key.getAttribute("data-note");
    console.log('keyNote:'+keyNote)

    key.classList.add("playing");
    note.innerHTML = keyNote;

}

function removeTransition(e) {
  if (e.propertyName !== "transform") return;
  this.classList.remove("playing");
}

function hintsOn(e, index) {
  e.setAttribute("style", "transition-delay:" + index * 50 + "ms");
}


keys.forEach(key => key.addEventListener("transitionend", removeTransition));

window.addEventListener("keydown", playNote);
