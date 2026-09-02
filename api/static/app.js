const form = document.getElementById('searchForm');
const input = document.getElementById('playerName');
const predictBtn = document.getElementById('predictBtn');
const loading = document.getElementById('loading');
const result = document.getElementById('result');

// The form navigates to the results page, so we just show a loading state and
// hand off to a POST. Pressing Enter inside the form fires this same handler
// natively - no extra keypress listener needed (that caused a double submit).
form.addEventListener('submit', (e) => {
    e.preventDefault();

    const playerName = input.value.trim();
    if (!playerName) return;

    predictBtn.disabled = true;
    loading.style.display = 'block';
    result.style.display = 'none';

    const hiddenForm = document.createElement('form');
    hiddenForm.method = 'POST';
    hiddenForm.action = '/api/predict';

    const hiddenInput = document.createElement('input');
    hiddenInput.type = 'hidden';
    hiddenInput.name = 'player_name';
    hiddenInput.value = playerName;

    hiddenForm.appendChild(hiddenInput);
    document.body.appendChild(hiddenForm);
    hiddenForm.submit();
});
