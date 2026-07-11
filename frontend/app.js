async function fetchAgents() {
    const response = await fetch("/api/agents");
    const agents = await response.json();

    const select = document.getElementById("agent-select");
    for (const agent of agents) {
        const option = document.createElement("option");
        option.value = agent.id;
        option.textContent = agent.name;
        select.appendChild(option);
    }
}

const canvas = document.getElementById("grid-canvas");
const ctx = canvas.getContext("2d");

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function renderFrame(trajectory, explored, frameIndex) {
    const cellSize = canvas.width / trajectory.grid_width;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (let y = 0; y < trajectory.grid_height; y++) {
        for (let x = 0; x < trajectory.grid_width; x++) {
            ctx.fillStyle = explored.has(`${x},${y}`) ? "#3a3a4a" : "#111";
            ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
        }
    }

    const frame = trajectory.frames[frameIndex];
    const warrior = frame.warriors[0];
    ctx.fillStyle = "#3b82f6";
    ctx.beginPath();
    ctx.arc(
        (warrior.x + 0.5) * cellSize,
        (warrior.y + 0.5) * cellSize,
        cellSize * 0.35,
        0,
        Math.PI * 2
    );
    ctx.fill();
}

async function playTrajectory(trajectory) {
    const explored = new Set();
    const stats = document.getElementById("stats");

    for (let i = 0; i < trajectory.frames.length; i++) {
        const frame = trajectory.frames[i];
        for (const [x, y] of frame.newly_revealed) {
            explored.add(`${x},${y}`);
        }

        renderFrame(trajectory, explored, i);
        stats.textContent = `Step: ${frame.step}/${trajectory.max_steps} | Explored: ${frame.explored_count}/${trajectory.total_tiles}`;

        await sleep(60);
    }

    stats.textContent =
        `Steps: ${trajectory.total_steps}/${trajectory.max_steps} | ` +
        `Explored: ${trajectory.total_explored}/${trajectory.total_tiles} | ` +
        (trajectory.success ? "Success" : "Failed");
}

async function runEpisode() {
    const agentId = document.getElementById("agent-select").value;
    const button = document.getElementById("run-btn");

    button.disabled = true;
    button.textContent = "Running...";

    const response = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agent_id: agentId }),
    });
    const trajectory = await response.json();

    await playTrajectory(trajectory);

    button.disabled = false;
    button.textContent = "Run";
}

document.getElementById("run-btn").addEventListener("click", runEpisode);

fetchAgents();
