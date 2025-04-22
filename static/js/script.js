document.addEventListener("DOMContentLoaded", () => {
  const fileForm = document.getElementById("fileForm");
  const spinner = document.getElementById("spinner");
  const resultDiv = document.getElementById("result");
  const resultText = document.getElementById("resultText");
  const ctx = document.getElementById("confidenceChart").getContext("2d");
  let chart;

  function showResult(data) {
    if (data.error) {
      alert("Error: " + data.error);
      return;
    }
    resultText.textContent = `Prediction: ${data.label}`;
    const conf = data.confidence;
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
      type: "bar",
      data: {
        labels: ["Rock", "Mine"],
        datasets: [
          {
            label: "Confidence",
            data: [conf.rock, conf.mine],
            backgroundColor: ["#3b82f6", "#f97316"],
          },
        ],
      },
      options: {
        scales: { y: { beginAtZero: true, max: 1 } },
        plugins: { legend: { display: false } },
      },
    });
    resultDiv.style.display = "block";
  }

  async function handleResponse(resPromise) {
    spinner.style.display = "block";
    resultDiv.style.display = "none";
    try {
      const res = await resPromise;
      const text = await res.text();
      try {
        const data = JSON.parse(text);
        showResult(data);
      } catch (err) {
        alert("Error parsing JSON: " + err.message + "\nResponse: " + text);
      }
    } catch (err) {
      alert("Network error: " + err.message);
    } finally {
      spinner.style.display = "none";
    }
  }

  fileForm.addEventListener("submit", (e) => {
    e.preventDefault();
    const formData = new FormData(fileForm);
    handleResponse(
      fetch("/predict-file", {
        method: "POST",
        body: formData,
      })
    );
  });
});
