const form = document.getElementById('searchForm');
        const input = document.getElementById('playerName');
        const clearBtn = document.getElementById('clearBtn');
        const predictBtn = document.getElementById('predictBtn');
        const loading = document.getElementById('loading');
        const result = document.getElementById('result');

        // Show/hide clear button
        input.addEventListener('input', () => {
            clearBtn.classList.toggle('active', input.value.length > 0);
        });

        // Clear input
        clearBtn.addEventListener('click', () => {
            input.value = '';
            clearBtn.classList.remove('active');
            input.focus();
            result.style.display = 'none';
        });

        // Form submission
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const playerName = input.value.trim();
            if (!playerName) return;

            // Show loading state
            predictBtn.disabled = true;
            loading.style.display = 'block';
            result.style.display = 'none';

            try {
                // Create a hidden form and submit it to trigger page navigation
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

            } catch (error) {
                result.innerHTML = `<strong>Error:</strong> ${error.message}`;
                result.style.display = 'block';
                loading.style.display = 'none';
                predictBtn.disabled = false;
            }
        });

        //Enter key implementation
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                form.dispatchEvent(new Event('submit'));
            }
        });