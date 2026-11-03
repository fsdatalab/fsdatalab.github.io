// Links the views of the key-theme figure: clicking a card or a bar
// highlights the same theme in all views. Clicking it again clears it.
document.querySelectorAll('[data-theme-dashboard]').forEach(function (root) {
  var items = root.querySelectorAll('[data-i]');
  var current = null;
  items.forEach(function (el) {
    el.addEventListener('click', function () {
      var i = el.getAttribute('data-i');
      current = current === i ? null : i;
      items.forEach(function (other) {
        other.classList.toggle('is-active', other.getAttribute('data-i') === current);
      });
    });
  });
});
