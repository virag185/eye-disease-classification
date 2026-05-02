(function () {
  "use strict";

  const model = window.RETINASCAN_MODEL;
  const classColors = {
    normal: "#16a34a",
    diabetic_retinopathy: "#dc2626",
    cataract: "#d97706",
    glaucoma: "#7c3aed"
  };
  const advice = {
    Normal: {
      icon: "OK",
      bg: "#f0fdf4",
      iconColor: "#16a34a",
      title: "No disease detected",
      sub: "Retinal image appears closest to the healthy class",
      steps: [
        "Continue annual eye examinations with your optometrist.",
        "Maintain a diet rich in Vitamins A, C, and E for eye health.",
        "Wear UV-protective sunglasses when outdoors.",
        "Report any sudden vision changes to your doctor promptly."
      ]
    },
    "Diabetic Retinopathy": {
      icon: "!",
      bg: "#fef2f2",
      iconColor: "#dc2626",
      title: "Diabetic retinopathy pattern detected",
      sub: "Ophthalmologist consultation is advised",
      steps: [
        "Consult an ophthalmologist for a comprehensive dilated eye exam.",
        "Maintain strict blood sugar control as recommended by your clinician.",
        "Control blood pressure and cholesterol, since both can worsen progression.",
        "Anti-VEGF injections or laser therapy may be recommended after clinical review.",
        "Schedule follow-ups every 3 to 6 months as directed by your doctor."
      ]
    },
    Cataract: {
      icon: "O",
      bg: "#fffbeb",
      iconColor: "#d97706",
      title: "Cataract pattern detected",
      sub: "Treatable lens opacity pattern",
      steps: [
        "Book a slit-lamp evaluation with an ophthalmologist.",
        "Cataract surgery is commonly performed and highly effective when clinically indicated.",
        "Use brighter lighting and anti-glare lenses in the interim.",
        "Avoid prolonged UV exposure, which can accelerate lens clouding.",
        "Post-surgery visual recovery is often rapid with good prognosis."
      ]
    },
    Glaucoma: {
      icon: "A",
      bg: "#f5f3ff",
      iconColor: "#7c3aed",
      title: "Glaucoma pattern suspected",
      sub: "Early treatment is critical to prevent vision loss",
      steps: [
        "Seek ophthalmological assessment including tonometry or eye-pressure testing.",
        "Visual field tests and OCT nerve-fiber analysis may be needed.",
        "Intraocular pressure-lowering drops are often first-line treatment.",
        "Avoid activities that significantly raise eye pressure until reviewed.",
        "Regular monitoring is essential to track progression."
      ]
    }
  };
  const risk = {
    Normal: "Low",
    Cataract: "Medium",
    "Diabetic Retinopathy": "High",
    Glaucoma: "High"
  };

  const els = {
    modelTag: document.getElementById("modelTag"),
    statAccuracy: document.getElementById("statAccuracy"),
    statImages: document.getElementById("statImages"),
    classAtlas: document.getElementById("classAtlas"),
    dropZone: document.getElementById("dropZone"),
    fileInput: document.getElementById("fileInput"),
    thumbRow: document.getElementById("thumbRow"),
    sampleRail: document.getElementById("sampleRail"),
    btnAnalyze: document.getElementById("btnAnalyze"),
    uploadCard: document.getElementById("uploadCard"),
    spinnerWrap: document.getElementById("spinnerWrap"),
    results: document.getElementById("results"),
    previewImage: document.getElementById("previewImage"),
    previewPlaceholder: document.getElementById("previewPlaceholder"),
    fileMeta: document.getElementById("fileMeta"),
    resClass: document.getElementById("resClass"),
    resConf: document.getElementById("resConf"),
    riskPill: document.getElementById("riskPill"),
    riskTxt: document.getElementById("riskTxt"),
    chipTime: document.getElementById("chipTime"),
    chipQuality: document.getElementById("chipQuality"),
    probBars: document.getElementById("probBars"),
    scanDetails: document.getElementById("scanDetails"),
    referenceStrip: document.getElementById("referenceStrip"),
    adviceCard: document.getElementById("adviceCard"),
    advIconWrap: document.getElementById("advIconWrap"),
    advIcon: document.getElementById("advIcon"),
    advTitle: document.getElementById("advTitle"),
    advSub: document.getElementById("advSub"),
    advSteps: document.getElementById("advSteps"),
    btnReset: document.getElementById("btnReset"),
    toast: document.getElementById("toast")
  };

  let files = [];
  let selectedIndex = 0;
  let selectedSample = null;
  let objectUrls = [];
  let previewUrl = null;
  let activeMeta = null;

  function classLabel(cls) {
    if (cls === "diabetic_retinopathy") return "Diabetic Retinopathy";
    return cls.replace(/_/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
  }

  function setupStats() {
    if (!model) return;
    els.modelTag.textContent = `${model.version || "RetinaForge"} API`;
    if (model.validation && typeof model.validation.accuracy === "number") {
      els.statAccuracy.textContent = `${Math.round(model.validation.accuracy * 100)}%`;
    }
    if (model.counts) {
      const total = Object.values(model.counts).reduce((a, b) => a + b, 0);
      els.statImages.textContent = total >= 1000 ? `${Math.round(total / 1000)}K+` : String(total);
    }
  }

  function formatConfidence(value) {
    if (value >= 0.995) return ">99%";
    return `${(value * 100).toFixed(1)}%`;
  }

  function firstSampleFor(cls) {
    return model && model.samples && model.samples[cls] && model.samples[cls][0]
      ? model.samples[cls][0]
      : "";
  }

  function renderClassAtlas() {
    if (!model || !els.classAtlas) return;
    const classNotes = {
      normal: "Healthy-looking retinal pattern",
      diabetic_retinopathy: "Retinal lesion and vascular risk pattern",
      cataract: "Cloudy lens opacity pattern",
      glaucoma: "Optic-disc risk pattern"
    };
    els.classAtlas.innerHTML = "";
    model.classes.forEach((cls) => {
      const item = document.createElement("article");
      item.className = "atlas-card";
      const sample = firstSampleFor(cls);
      item.innerHTML = `
        <img alt="${classLabel(cls)} reference retina" src="../${sample}">
        <div>
          <strong>${classLabel(cls)}</strong>
          <span>${classNotes[cls] || "Dataset reference class"}</span>
        </div>
      `;
      els.classAtlas.appendChild(item);
    });
  }

  function sampleItems() {
    if (!model || !model.samples) return [];
    return model.classes.flatMap((cls) =>
      (model.samples[cls] || []).slice(0, 2).map((src, index) => ({
        cls,
        src,
        label: `${classLabel(cls)} ${index + 1}`
      }))
    );
  }

  function renderSampleRail() {
    els.sampleRail.innerHTML = "";
    sampleItems().forEach((item) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sample-chip";
      button.classList.toggle("active", selectedSample && selectedSample.src === item.src);
      button.innerHTML = `
        <img alt="${item.label}" src="../${item.src}">
        <span>${classLabel(item.cls)}</span>
      `;
      button.addEventListener("click", () => selectSample(item));
      els.sampleRail.appendChild(button);
    });
  }

  function setPreview(src, title, subtitle) {
    els.previewImage.src = src;
    els.previewImage.classList.add("show");
    els.previewPlaceholder.classList.add("hide");
    els.fileMeta.innerHTML = `<strong>${title}</strong><span>${subtitle}</span>`;
  }

  function clearPreview() {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    els.previewImage.removeAttribute("src");
    els.previewImage.classList.remove("show");
    els.previewPlaceholder.classList.remove("hide");
    els.fileMeta.innerHTML = "<strong>Ready for scan</strong><span>Select an upload or dataset sample to begin.</span>";
  }

  function selectSample(item) {
    selectedSample = item;
    files = [];
    selectedIndex = 0;
    renderThumbs();
    renderSampleRail();
    activeMeta = {
      source: "Dataset sample",
      name: item.label,
      size: "Built-in dataset image",
      type: classLabel(item.cls)
    };
    setPreview(`../${item.src}`, item.label, "Dataset sample loaded. Click Analyze Image.");
    els.btnAnalyze.disabled = false;
    toast(`${item.label} loaded`);
  }

  function addFiles(nextFiles) {
    const imageFiles = nextFiles.filter((file) => file.type.startsWith("image/"));
    if (!imageFiles.length) {
      toast("Please choose an image file.");
      return;
    }
    selectedSample = null;
    files = [...files, ...imageFiles].slice(0, 8);
    selectedIndex = Math.min(selectedIndex, Math.max(0, files.length - 1));
    renderThumbs();
    renderSampleRail();
    els.btnAnalyze.disabled = !files.length;
    updateFilePreview();
  }

  function removeFile(index) {
    files.splice(index, 1);
    if (selectedIndex >= files.length) {
      selectedIndex = Math.max(0, files.length - 1);
    }
    renderThumbs();
    els.btnAnalyze.disabled = !files.length && !selectedSample;
    if (files.length) {
      updateFilePreview();
    } else if (!selectedSample) {
      clearPreview();
    }
  }

  function updateFilePreview() {
    if (!files[selectedIndex]) return;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    const file = files[selectedIndex];
    previewUrl = URL.createObjectURL(file);
    const sizeKb = Math.max(1, Math.round(file.size / 1024));
    activeMeta = {
      source: "Uploaded image",
      name: file.name,
      size: `${sizeKb} KB`,
      type: file.type || "image file"
    };
    setPreview(previewUrl, file.name, `${sizeKb} KB selected. You can add up to 8 images.`);
  }

  function renderThumbs() {
    objectUrls.forEach((url) => URL.revokeObjectURL(url));
    objectUrls = [];
    els.thumbRow.innerHTML = "";
    if (!files.length) {
      els.thumbRow.classList.remove("show");
      return;
    }

    els.thumbRow.classList.add("show");
    files.forEach((file, index) => {
      const url = URL.createObjectURL(file);
      objectUrls.push(url);
      const item = document.createElement("div");
      item.className = `thumb${index === selectedIndex ? " active" : ""}`;
      const image = document.createElement("img");
      image.alt = file.name;
      image.src = url;
      const remove = document.createElement("div");
      remove.className = "thumb-x";
      remove.textContent = "x";
      remove.addEventListener("click", (event) => {
        event.stopPropagation();
        removeFile(index);
      });
      item.addEventListener("click", () => {
        selectedIndex = index;
        updateFilePreview();
        renderThumbs();
      });
      item.append(image, remove);
      els.thumbRow.appendChild(item);
    });
  }

  async function analyzeSelected() {
    if (!files[selectedIndex] && !selectedSample) return;
    els.uploadCard.style.display = "none";
    els.spinnerWrap.style.display = "block";
    els.results.style.display = "none";
    const started = performance.now();
    try {
      let response;
      if (selectedSample) {
        response = await fetch("/api/analyze-sample", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path: selectedSample.src })
        });
      } else {
        const form = new FormData();
        form.append("image", files[selectedIndex]);
        response = await fetch("/api/analyze", {
          method: "POST",
          body: form
        });
      }
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.error || "Backend analysis failed.");
      }
      const payload = await response.json();
      renderResults(payload, performance.now() - started);
    } catch (error) {
      els.uploadCard.style.display = "block";
      toast(error.message || "Backend is not connected.");
    } finally {
      els.spinnerWrap.style.display = "none";
    }
  }

  function renderResults(payload, elapsedMs) {
    const top = payload.ranks[0];
    const label = classLabel(top.cls);
    const color = classColors[top.cls] || "#0ea5a0";
    const riskLevel = risk[label] || (label === "Normal" ? "Low" : "High");

    els.results.style.display = "block";
    els.resClass.textContent = label;
    els.resClass.style.color = color;
    els.resConf.textContent = formatConfidence(top.probability);
    els.chipTime.textContent = `${Math.max(1, Math.round(elapsedMs))}ms inference`;
    if (payload.quality) {
      els.chipQuality.textContent = `Sharpness ${Math.round(payload.quality.sharpness)}/100`;
    }

    els.riskPill.className = `risk-pill risk-${riskLevel.toLowerCase()}`;
    els.riskTxt.textContent = `${riskLevel} Risk`;

    els.probBars.innerHTML = "";
    payload.ranks.forEach((rank) => {
      const rankLabel = classLabel(rank.cls);
      const rankColor = classColors[rank.cls] || "#0ea5a0";
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-head">
          <div class="prob-name" style="color:${rankColor}">${rankLabel}</div>
          <div class="prob-pct">${formatConfidence(rank.probability)}</div>
        </div>
        <div class="prob-track">
          <div class="prob-fill" style="width:${(rank.probability * 100).toFixed(1)}%;background:${rankColor}"></div>
        </div>
      `;
      els.probBars.appendChild(row);
    });

    renderAdvice(label);
    renderScanDetails(payload, label);
    renderReferences(top.cls);
    window.scrollTo({ top: els.results.offsetTop - 24, behavior: "smooth" });
  }

  function renderScanDetails(payload, label) {
    const quality = payload.quality || {};
    const rows = [
      ["Source", activeMeta ? activeMeta.source : "Selected image"],
      ["File", activeMeta ? activeMeta.name : "Retinal image"],
      ["Type", activeMeta ? activeMeta.type : "Image"],
      ["Size", activeMeta ? activeMeta.size : "Not available"],
      ["Primary output", label],
      ["Brightness", quality.brightness !== undefined ? `${Math.round(quality.brightness)}/100` : "Not available"],
      ["Contrast", quality.contrast !== undefined ? `${Math.round(quality.contrast)}/100` : "Not available"],
      ["Sharpness", quality.sharpness !== undefined ? `${Math.round(quality.sharpness)}/100` : "Not available"]
    ];
    els.scanDetails.innerHTML = rows.map(([key, value]) => `
      <div class="detail-row"><span>${key}</span><strong>${value}</strong></div>
    `).join("");
  }

  function renderReferences(cls) {
    if (!model || !model.samples || !model.samples[cls]) return;
    els.referenceStrip.innerHTML = "";
    model.samples[cls].slice(0, 4).forEach((src, index) => {
      const item = document.createElement("div");
      item.className = "reference-image";
      item.innerHTML = `
        <img alt="${classLabel(cls)} reference ${index + 1}" src="../${src}">
        <span>${classLabel(cls)} ref ${index + 1}</span>
      `;
      els.referenceStrip.appendChild(item);
    });
  }

  function renderAdvice(label) {
    const info = advice[label];
    if (!info) {
      els.adviceCard.style.display = "none";
      return;
    }
    els.advIconWrap.style.background = info.bg;
    els.advIcon.textContent = info.icon;
    els.advIcon.style.color = info.iconColor;
    els.advTitle.textContent = info.title;
    els.advSub.textContent = info.sub;
    els.advSteps.innerHTML = "";
    info.steps.forEach((step) => {
      const item = document.createElement("li");
      item.textContent = step;
      els.advSteps.appendChild(item);
    });
    els.adviceCard.style.display = "block";
  }

  function reset() {
    files = [];
    selectedIndex = 0;
    selectedSample = null;
    activeMeta = null;
    renderThumbs();
    renderSampleRail();
    clearPreview();
    els.fileInput.value = "";
    els.btnAnalyze.disabled = true;
    els.results.style.display = "none";
    els.adviceCard.style.display = "none";
    els.uploadCard.style.display = "block";
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function toast(message) {
    els.toast.textContent = message;
    els.toast.classList.add("show");
    setTimeout(() => els.toast.classList.remove("show"), 3200);
  }

  function bindEvents() {
    els.fileInput.addEventListener("change", (event) => addFiles(Array.from(event.target.files)));
    els.dropZone.addEventListener("dragover", (event) => {
      event.preventDefault();
      els.dropZone.classList.add("over");
    });
    els.dropZone.addEventListener("dragleave", () => els.dropZone.classList.remove("over"));
    els.dropZone.addEventListener("drop", (event) => {
      event.preventDefault();
      els.dropZone.classList.remove("over");
      addFiles(Array.from(event.dataTransfer.files));
    });
    els.btnAnalyze.addEventListener("click", analyzeSelected);
    els.btnReset.addEventListener("click", reset);
  }

  setupStats();
  renderClassAtlas();
  renderSampleRail();
  bindEvents();
})();
