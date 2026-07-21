const COLORS = {
  positive: "#4ade80",
  neutral: "#94a3b8",
  negative: "#f87171",
  accent: "#6ea8fe",
  accent2: "#c084fc",
  accent3: "#fbbf24",
};

async function loadJSON(path) {
  const res = await fetch(path);
  return res.json();
}

function fmt(n) {
  return new Intl.NumberFormat("en-US").format(n);
}

function renderStats({ topics, impact }) {
  const totalArticles = topics.reduce((sum, t) => sum + t.topic_size, 0);
  const totalIndustries = new Set(topics.map((t) => t.industry)).size;
  const stats = [
    { num: fmt(totalArticles), label: "Curated articles" },
    { num: topics.length, label: "Final topics" },
    { num: totalIndustries, label: "Industries covered" },
  ];
  const el = document.getElementById("stats");
  el.innerHTML = stats
    .map((s) => `<div class="stat"><div class="num">${s.num}</div><div class="label">${s.label}</div></div>`)
    .join("");
}

function renderSentimentTime(data) {
  const dates = [...new Set(data.map((d) => d.date))].sort();
  const bySentiment = (label) =>
    dates.map((d) => {
      const row = data.find((r) => r.date === d && r.sentiment === label);
      return row ? +(row.pct * 100).toFixed(1) : null;
    });

  new Chart(document.getElementById("sentimentTimeChart"), {
    type: "line",
    data: {
      labels: dates,
      datasets: [
        { label: "Positive", data: bySentiment("positive"), borderColor: COLORS.positive, backgroundColor: COLORS.positive, tension: 0.25, pointRadius: 0 },
        { label: "Neutral", data: bySentiment("neutral"), borderColor: COLORS.neutral, backgroundColor: COLORS.neutral, tension: 0.25, pointRadius: 0 },
        { label: "Negative", data: bySentiment("negative"), borderColor: COLORS.negative, backgroundColor: COLORS.negative, tension: 0.25, pointRadius: 0 },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ticks: { color: "#8b91a1", maxTicksLimit: 12 }, grid: { color: "#232733" } },
        y: { ticks: { color: "#8b91a1", callback: (v) => v + "%" }, grid: { color: "#232733" } },
      },
      plugins: { legend: { labels: { color: "#e8eaef" } } },
    },
  });
}

function renderTopics(topics) {
  new Chart(document.getElementById("topicChart"), {
    type: "bar",
    data: {
      labels: topics.map((t) => t.final_topic),
      datasets: [{ label: "Articles", data: topics.map((t) => t.topic_size), backgroundColor: COLORS.accent }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#8b91a1" }, grid: { color: "#232733" } },
        y: { ticks: { color: "#e8eaef" }, grid: { display: false } },
      },
      plugins: { legend: { display: false } },
    },
  });

  const tbody = document.querySelector("#topicTable tbody");
  tbody.innerHTML = topics
    .map(
      (t) => `<tr>
        <td><strong>${t.final_topic}</strong><br><span class="tag">${t.keywords.split(",").slice(0, 4).join(", ")}</span></td>
        <td>${t.industry}</td>
        <td>${t.technology}</td>
        <td>${t.impact_mode}</td>
        <td>${fmt(t.topic_size)}</td>
      </tr>`
    )
    .join("");
}

function topPerIndustry(rows, labelKey) {
  const byIndustry = {};
  for (const r of rows) {
    if (!byIndustry[r.industry_label] || r.count > byIndustry[r.industry_label].count) {
      byIndustry[r.industry_label] = r;
    }
  }
  return Object.values(byIndustry).sort((a, b) => b.count - a.count).slice(0, 10);
}

function renderCategoryChart(canvasId, rows, categoryKey, palette) {
  const top = topPerIndustry(rows, categoryKey);
  const categories = [...new Set(top.map((r) => r[categoryKey]))];
  const colorFor = (cat) => palette[categories.indexOf(cat) % palette.length];

  new Chart(document.getElementById(canvasId), {
    type: "bar",
    data: {
      labels: top.map((r) => r.industry_label),
      datasets: [
        {
          label: "Mentions",
          data: top.map((r) => r.count),
          backgroundColor: top.map((r) => colorFor(r[categoryKey])),
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { ticks: { color: "#8b91a1" }, grid: { color: "#232733" } },
        y: { ticks: { color: "#e8eaef" }, grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `${top[ctx.dataIndex][categoryKey]}: ${fmt(ctx.raw)} mentions`,
          },
        },
      },
    },
  });
}

async function main() {
  const [sentimentTime, topics, impact, tech] = await Promise.all([
    loadJSON("./data/sentiment_over_time.json"),
    loadJSON("./data/topics.json"),
    loadJSON("./data/industry_impact_counts.json"),
    loadJSON("./data/industry_technology_counts.json"),
  ]);

  renderStats({ topics, impact });
  renderSentimentTime(sentimentTime);
  renderTopics(topics);
  renderCategoryChart("impactChart", impact, "impact_modes", [COLORS.accent, COLORS.accent2, COLORS.accent3, COLORS.positive, COLORS.negative, COLORS.neutral]);
  renderCategoryChart("techChart", tech, "technologies", [COLORS.accent2, COLORS.accent, COLORS.accent3, COLORS.positive, COLORS.negative, COLORS.neutral]);
}

main();
