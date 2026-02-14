// DOM Elements
const compileBtn = document.getElementById("compileBtn");
const outputPanel = document.getElementById("outputPanel");
const closeBtn = document.getElementById("closeBtn");
const tabButtons = document.querySelectorAll(".tab-btn");
const tabContents = document.querySelectorAll(".tab-content");

// Hide panel initially
outputPanel.classList.remove("show");

// Format tokens like terminal output
function formatTokens(tokens) {
    if (!tokens || tokens.length === 0) {
        return "No Data";
    }
    
    // Filter out EOF token for display consistency with terminal
    const displayTokens = tokens.filter(token => token.type !== 'EOF');
    
    return displayTokens.map(token => {
        return `  ${token.type} -> ${token.value} (line ${token.lineno})`;
    }).join('\n');
}

// Format TAC instructions for consistent display
function formatTAC(tac) {
    if (!tac || tac.length === 0) {
        return "No Data";
    }
    return tac.join('\n');
}

// Compile button click
compileBtn.addEventListener("click", async () => {
    outputPanel.classList.add("show");
    const code = document.getElementById("code").value;

    try {
        const response = await fetch("http://127.0.0.1:5000/compile", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ code })
        });

        const result = await response.json();

        // Display data with proper formatting
        document.getElementById("tokens").innerText = result.tokens && result.tokens.length ? formatTokens(result.tokens) : "No Data";
        document.getElementById("ast").innerText = result.ast && Object.keys(result.ast).length ? JSON.stringify(result.ast, null, 2) : "No Data";
        document.getElementById("tac").innerText = result.tac && result.tac.length ? formatTAC(result.tac) : "No Data";
        document.getElementById("optimized_tac").innerText = result.optimized_tac && result.optimized_tac.length ? formatTAC(result.optimized_tac) : "No Data";
        document.getElementById("assembly").innerText = result.assembly && result.assembly.length ? result.assembly.join("\n") : "No Data";
        document.getElementById("output").innerText = result.output ? result.output : "No Data";
        document.getElementById("errors").innerText = result.errors && result.errors.length ? result.errors.join("\n") : "No Data";

        // Reset tabs to first tab
        tabButtons.forEach(btn => btn.classList.remove("active"));
        tabContents.forEach(tc => tc.classList.remove("active"));
        tabButtons[0].classList.add("active");
        tabContents[0].classList.add("active");

    } catch (err) {
        console.error("Fetch failed:", err);
        document.getElementById("errors").innerText = "Error connecting to backend!";
    }
});

// Close button
closeBtn.addEventListener("click", () => {
    outputPanel.classList.remove("show");
});

// Tab switching
tabButtons.forEach(btn => {
    btn.addEventListener("click", () => {
        const tab = btn.dataset.tab;
        tabButtons.forEach(b => b.classList.remove("active"));
        tabContents.forEach(tc => tc.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById(tab).classList.add("active");
    });
});