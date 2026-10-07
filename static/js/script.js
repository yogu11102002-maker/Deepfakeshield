// Show / Hide Password (login page)

const togglePasswordIcon = document.getElementById("togglePassword");
const password = document.getElementById("password");

if (togglePasswordIcon && password) {

    togglePasswordIcon.addEventListener("click", function () {

        if (password.type === "password") {

            password.type = "text";

            this.classList.remove("fa-eye");
            this.classList.add("fa-eye-slash");

        } else {

            password.type = "password";

            this.classList.remove("fa-eye-slash");
            this.classList.add("fa-eye");

        }

    });

    togglePasswordIcon.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            togglePasswordIcon.click();
        }
    });

}


// Login — calls backend, creates session, redirects to dashboard

const loginForm = document.getElementById("loginForm");

if (loginForm) {

    loginForm.addEventListener("submit", async function (event) {

        event.preventDefault();

        const email = document.getElementById("email").value;
        const password = document.getElementById("password").value;

        try {

            const response = await fetch("/api/login", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ email, password })
            });

            const data = await response.json();

            if (data.success) {
                window.location.href = "/dashboard";
            } else {
                alert(data.message || "Login failed.");
            }

        } catch (err) {
            alert("Something went wrong. Please try again.");
        }

    });

}


// Show / Hide Password (signup page - reusable function)

function togglePassword(id, icon) {

    const input = document.getElementById(id);

    if (input.type === "password") {

        input.type = "text";

        icon.classList.remove("fa-eye");
        icon.classList.add("fa-eye-slash");

    } else {

        input.type = "password";

        icon.classList.remove("fa-eye-slash");
        icon.classList.add("fa-eye");

    }

}


// Signup — saves the new user in the database via backend

const signupForm = document.getElementById("signupForm");

if (signupForm) {

    signupForm.addEventListener("submit", async function (event) {

        event.preventDefault();

        const name = document.getElementById("name").value;
        const email = document.getElementById("signupEmail").value;
        const password = document.getElementById("signupPassword").value;
        const confirmPassword = document.getElementById("confirmPassword").value;

        if (password !== confirmPassword) {
            alert("Passwords do not match.");
            return;
        }

        try {

            const response = await fetch("/api/register", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name, email, password, confirmPassword })
            });

            const data = await response.json();

            if (data.success) {
                alert(data.message || "Account created. Check your email to verify it.");
                window.location.href = "/login";
            } else {
                alert(data.message || "Registration failed.");
            }

        } catch (err) {
            alert("Something went wrong. Please try again.");
        }

    });

}


// Dashboard file upload

const fileInput = document.getElementById("fileInput");
const fileName = document.getElementById("fileName");

if (fileInput && fileName) {

    fileInput.addEventListener("change", function () {

        if (this.files.length > 0) {
            fileName.textContent = "Selected: " + this.files[0].name;

            // clear any text input so only one input is analyzed at a time
            const textBox = document.getElementById("textInput");
            if (textBox) textBox.value = "";
        }

    });

}


// Dashboard — run analysis (image or text) via the Hugging Face-backed API
// Supports the dashboard's quick-upload widget (id="fileInput") and the
// dedicated Analyze Content page's tabs (id="imageInput"/"videoInput"/"audioInput").

const analyzeBtn = document.getElementById("analyzeBtn");

if (analyzeBtn) {

    analyzeBtn.addEventListener("click", async function () {

        const candidateInputs = ["fileInput", "imageInput", "videoInput", "audioInput"]
            .map(id => document.getElementById(id))
            .filter(el => el && el.files && el.files.length > 0);

        const file = candidateInputs.length > 0 ? candidateInputs[0].files[0] : null;
        const textBox = document.getElementById("textInput");
        const text = textBox ? textBox.value.trim() : "";

        const resultPanel = document.getElementById("resultPanel");
        const resultIcon = document.getElementById("resultIcon");
        const resultLabel = document.getElementById("resultLabel");
        const resultConfidence = document.getElementById("resultConfidence");
        const resultMessage = document.getElementById("resultMessage");

        if (!file && !text) {
            alert("Please upload an image or paste some text first.");
            return;
        }

        const formData = new FormData();
        if (file) {
            formData.append("file", file);
        } else {
            formData.append("text", text);
        }

        const originalBtnHTML = analyzeBtn.innerHTML;
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...';

        try {

            const response = await fetch("/api/analyze", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!data.success) {
                alert(data.message || "Analysis failed. Please try again.");
                return;
            }

            resultPanel.hidden = false;

            const status = typeof data.status === "string" ? data.status : "uncertain";
            if (status === 'failed') {
                resultPanel.className = "result-panel result-failed";
                resultIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
            } else if (status === 'conflict') {
                resultPanel.className = "result-panel result-conflict";
                resultIcon.innerHTML = '<i class="fa-solid fa-code-compare"></i>';
            } else if (status === 'uncertain' || status === 'inconclusive') {
                resultPanel.className = "result-panel result-uncertain";
                resultIcon.innerHTML = '<i class="fa-solid fa-circle-question"></i>';
            } else if (data.is_threat) {
                resultPanel.className = "result-panel result-threat";
                resultIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
            } else {
                resultPanel.className = "result-panel result-safe";
                resultIcon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
            }

            const safeText = value => typeof value === "string"
                ? value.replace(/reality[\s_-]*defender/gi, "external detection service")
                : "";
            const contentType = safeText(data.content_type) || "Content";
            const label = safeText(data.label) || "Inconclusive";
            resultLabel.textContent = contentType.charAt(0).toUpperCase() + contentType.slice(1) + " result: " + label;
            const deepseek = data.evidence && data.evidence.deepseek_review;
            const deepseekText = deepseek && deepseek.status === "available"
                ? "Visual review: " + safeText(deepseek.opinion)
                : "";
            const hasScore = data.fake_score != null && Number.isFinite(Number(data.fake_score));
            const scoreText = hasScore
                ? "Detection score: " + (Number(data.fake_score) * 100).toFixed(1) + "%. "
                : "Detection score unavailable. ";
            const warnings = Array.isArray(data.warnings)
                ? data.warnings.map(safeText).filter(Boolean).join(" ")
                : "";
            const message = safeText(data.message);
            if (resultMessage) resultMessage.textContent = message;
            let videoText = "";
            if (data.content_type === "video") {
                const percent = value => value == null || !Number.isFinite(Number(value))
                    ? "N/A"
                    : (Number(value) * 100).toFixed(1) + "%";
                const frameErrors = Array.isArray(data.frame_errors)
                    ? data.frame_errors.map(item => safeText(item && item.error)).filter(Boolean)
                    : [];
                const details = [
                    "Video verdict: " + (safeText(data.verdict) || (status === "failed" ? "Analysis failed" : "Inconclusive")),
                    "Max frame score: " + percent(data.max_frame),
                    "Fake frames: " + (Number.isFinite(Number(data.fake_frame_count)) ? Number(data.fake_frame_count) : "N/A"),
                    "Audio score: " + percent(data.audio)
                ];
                if (safeText(data.audio_error)) details.push("Audio: " + safeText(data.audio_error));
                if (frameErrors.length) details.push("Frame errors: " + frameErrors.join("; "));
                videoText = details.join(". ") + ". ";
            }
            resultConfidence.textContent = "Detection engine: DeepFake Shield. " +
                (data.content_type === "video" ? videoText : scoreText) + warnings +
                (deepseekText ? " " + deepseekText : "");


        } catch (err) {
            alert("Something went wrong while analyzing. Please try again.");
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = originalBtnHTML;
        }

    });

}


// Logout — clears the server-side session

const logoutLink = document.getElementById("logoutLink");

if (logoutLink) {

    logoutLink.addEventListener("click", async function (event) {

        event.preventDefault();

        await fetch("/api/logout", { method: "POST" });

        window.location.href = "/login";

    });

}

// Profile form
const profileForm = document.getElementById('profileForm');
if (profileForm) {
    profileForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const name = document.getElementById('profileName').value;

        try {
            const response = await fetch('/api/update-profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });

            const data = await response.json();

            if (data.success) {
                alert('Profile updated successfully!');
                location.reload();
            } else {
                alert(data.message || 'Failed to update profile.');
            }
        } catch (err) {
            alert('Something went wrong.');
        }
    });
}

// Password form
const passwordForm = document.getElementById('passwordForm');
if (passwordForm) {
    passwordForm.addEventListener('submit', async function(e) {
        e.preventDefault();

        const oldPassword = document.getElementById('oldPassword').value;
        const newPassword = document.getElementById('newPassword').value;
        const confirmPassword = document.getElementById('confirmPassword').value;

        if (newPassword !== confirmPassword) {
            alert('New passwords do not match.');
            return;
        }

        try {
            const response = await fetch('/api/change-password', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ oldPassword, newPassword, confirmPassword })
            });

            const data = await response.json();

            if (data.success) {
                alert('Password changed successfully!');
                passwordForm.reset();
            } else {
                alert(data.message || 'Failed to change password.');
            }
        } catch (err) {
            alert('Something went wrong.');
        }
    });
}

// Delete account
function confirmDeleteAccount() {
    const confirmed = confirm('⚠️ Warning: This will permanently delete your account and all analysis records. This cannot be undone. Type "DELETE" to confirm.');
    
    if (!confirmed) return;

    const userInput = prompt('Type "DELETE" to confirm account deletion:');
    
    if (userInput === 'DELETE') {
        deleteAccount();
    } else {
        alert('Account deletion cancelled.');
    }
}

async function deleteAccount() {
    try {
        const response = await fetch('/api/delete-account', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const data = await response.json();

        if (data.success) {
            alert('Account deleted successfully. Redirecting to home...');
            window.location.href = '{{ url_for("home") }}';
        } else {
            alert(data.message || 'Failed to delete account.');
        }
    } catch (err) {
        alert('Something went wrong.');
    }
}