/* global tf, tfvis, DataIO, document, window */

(() => {
  // ---------- State ----------
  const state = {
    rawRatings: [],
    rawItems: [],
    maps: null,
    X: null, // { users, items, ratings }
    titlesByIdx: [],
    model: null,
    stopRequested: false,
    trained: false,
  };

  // ---------- UI Helpers ----------
  const $ = (id) => document.getElementById(id);

  const setSummary = (el, html) => { el.innerHTML = html; };
  const setTable = (el, rows, headers) => {
    if (!rows || rows.length === 0) { el.innerHTML = ''; return; }
    let thead = '<thead><tr>' + headers.map(h=>`<th>${h}</th>`).join('') + '</tr></thead>';
    let tbody = '<tbody>' + rows.map(r => `<tr>${r.map(c=>`<td>${c}</td>`).join('')}</tr>`).join('') + '</tbody>';
    el.innerHTML = thead + tbody;
  };

  const prettyNum = (x, d=4) => Number.isFinite(x) ? x.toFixed(d) : String(x);

  // ---------- File Reading ----------
  const readFileText = (file) => {
    return new Promise((resolve, reject) => {
      const fr = new FileReader();
      fr.onload = () => resolve(fr.result);
      fr.onerror = reject;
      fr.readAsText(file);
    });
  };

  // ---------- Model ----------
  class MFRecommender {
    constructor(nUsers, nItems, k=32, reg=1e-3) {
      this.nUsers = nUsers;
      this.nItems = nItems;
      this.k = k;
      this.reg = reg;

      // Parameters (Embedding "layers" as variables)
      this.userEmb = tf.variable(tf.randomNormal([nUsers, k], 0, 0.05, 'float32'));
      this.itemEmb = tf.variable(tf.randomNormal([nItems, k], 0, 0.05, 'float32'));
      this.userBias = tf.variable(tf.zeros([nUsers]));
      this.itemBias = tf.variable(tf.zeros([nItems]));
      this.globalBias = tf.variable(tf.scalar(0, 'float32'));

      this.optimizer = tf.train.adam();
    }

    predictBatch(userIdx, itemIdx) {
      return tf.tidy(() => {
        const uVecs = tf.gather(this.userEmb, userIdx); // [B, k]
        const iVecs = tf.gather(this.itemEmb, itemIdx); // [B, k]
        const dot = tf.sum(tf.mul(uVecs, iVecs), -1);   // [B]
        const ub = tf.gather(this.userBias, userIdx);
        const ib = tf.gather(this.itemBias, itemIdx);
        return tf.addN([dot, ub, ib, this.globalBias]); // [B]
      });
    }

    lossFn(pred, trueRatings) {
      return tf.tidy(() => {
        const mse = tf.losses.meanSquaredError(trueRatings, pred);
        const l2 = tf.add(tf.sum(tf.square(this.userEmb)), tf.sum(tf.square(this.itemEmb)));
        const regTerm = tf.mul(this.reg, l2);
        return tf.add(mse, regTerm);
      });
    }

    async train(dataset, params, onEpoch) {
      const { users, items, ratings } = dataset; // arrays
      const {
        epochs = 10, batchSize = 256, lr = 0.01, split = 0.8
      } = params;

      this.optimizer = tf.train.adam(lr);

      // Train/val split
      const N = ratings.length;
      const idx = Array.from({length:N}, (_,i)=>i);
      shuffleInPlace(idx);
      const nTrain = Math.floor(split * N);
      const trainIdx = idx.slice(0, nTrain);
      const valIdx = idx.slice(nTrain);

      const toTensor = (indices) => {
        const u = tf.tensor1d(indices.map(i => users[i]), 'int32');
        const it = tf.tensor1d(indices.map(i => items[i]), 'int32');
        const r = tf.tensor1d(indices.map(i => ratings[i]), 'float32');
        return {u, it, r};
      };

      const trainT = toTensor(trainIdx);
      const valT = toTensor(valIdx);

      // Estimate global bias as mean rating
      const meanRating = tf.mean(trainT.r).dataSync()[0];
      this.globalBias.assign(tf.scalar(meanRating, 'float32'));

      for (let epoch = 1; epoch <= epochs; epoch++) {
        if (state.stopRequested) break;
        const batches = makeBatches(trainIdx.length, batchSize);
        let epochLoss = 0;

        for (const [bStart, bEnd] of batches) {
          if (state.stopRequested) break;
          const bIdx = trainIdx.slice(bStart, bEnd);
          const bu = tf.tensor1d(bIdx.map(i => users[i]), 'int32');
          const bi = tf.tensor1d(bIdx.map(i => items[i]), 'int32');
          const br = tf.tensor1d(bIdx.map(i => ratings[i]), 'float32');

          const lossVal = this.optimizer.minimize(() => {
            const pred = this.predictBatch(bu, bi);
            return this.lossFn(pred, br);
          }, true);

          epochLoss += lossVal.dataSync()[0];
          tf.dispose([bu, bi, br, lossVal]);
          await tf.nextFrame(); // yield to UI
        }

        // Compute RMSE on train/val
        const trainRmse = await this.rmse(trainT);
        const valRmse = await this.rmse(valT);

        if (onEpoch) onEpoch({ epoch, epochs, trainRmse, valRmse, epochLoss });
      }

      // Cleanup
      tf.dispose([trainT.u, trainT.it, trainT.r, valT.u, valT.it, valT.r]);
    }

    async rmse({u, it, r}) {
      const pred = this.predictBatch(u, it);
      const mse = tf.metrics.meanSquaredError(r, pred);
      const rmse = tf.sqrt(mse);
      const val = (await rmse.data())[0];
      tf.dispose([pred, mse, rmse]);
      return val;
    }

    predictSingle(userIdx, itemIdx) {
      return tf.tidy(() => {
        const u = tf.tensor1d([userIdx], 'int32');
        const i = tf.tensor1d([itemIdx], 'int32');
        const p = this.predictBatch(u, i);
        const v = p.dataSync()[0];
        return v;
      });
    }

    // Recommend topN items for userIdx (excluding already-rated if provided set)
    recommendForUser(userIdx, topN, ratedSet) {
      return tf.tidy(() => {
        const uRow = tf.gather(this.userEmb, userIdx);      // [k]
        const dots = tf.matMul(this.itemEmb, uRow.expandDims(1)).squeeze(); // [nItems]
        const scores = tf.addN([dots, this.itemBias, this.globalBias]);     // [nItems]

        const arr = Array.from(scores.dataSync());
        const recs = [];
        for (let i = 0; i < arr.length; i++) {
          if (ratedSet && ratedSet.has(i)) continue;
          recs.push({ itemIdx: i, score: arr[i] });
        }
        recs.sort((a,b)=>b.score - a.score);
        return recs.slice(0, topN);
      });
    }

    // Accessors for Model Anatomy UI
    getUserVector(userIdx){
      return tf.tidy(() => Array.from(tf.gather(this.userEmb, tf.tensor1d([userIdx],'int32')).squeeze().dataSync()));
    }
    getItemVector(itemIdx){
      return tf.tidy(() => Array.from(tf.gather(this.itemEmb, tf.tensor1d([itemIdx],'int32')).squeeze().dataSync()));
    }
    getBiases(userIdx, itemIdx){
      return tf.tidy(() => {
        const ub = tf.gather(this.userBias, tf.tensor1d([userIdx],'int32')).dataSync()[0];
        const ib = tf.gather(this.itemBias, tf.tensor1d([itemIdx],'int32')).dataSync()[0];
        const mu = this.globalBias.dataSync()[0];
        return { ub, ib, mu };
      });
    }
  }

  // ---- Tensor save/load helpers (JSON) ----
  async function saveTensor(t, filename) {
    const obj = { shape: t.shape, dtype: t.dtype, data: Array.from(t.dataSync()) };
    const blob = new Blob([JSON.stringify(obj)], {type:'application/json'});
    triggerDownload(blob, filename);
  }
  async function loadTensor(filename) {
    const file = await pickFile(filename);
    const text = await file.text();
    const obj = JSON.parse(text);
    return tf.tensor(obj.data, obj.shape, obj.dtype);
  }
  function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
  }
  function pickFile() {
    return new Promise((resolve) => {
      const input = document.createElement('input');
      input.type = 'file'; input.accept = '.json';
      input.onchange = () => resolve(input.files[0]);
      input.click();
    });
  }

  // ---------- Utils ----------
  function shuffleInPlace(arr){
    for (let i=arr.length-1;i>0;i--){
      const j = (Math.random()* (i+1))|0;
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
  function makeBatches(n, b){
    const res = [];
    for (let i=0;i<n;i+=b) res.push([i, Math.min(n, i+b)]);
    return res;
  }
  function clamp(x, lo=1, hi=5){ return Math.max(lo, Math.min(hi, x)); }

  // Build already-rated set per user
  function buildRatedSet(X){
    const map = new Map(); // userIdx -> Set(itemIdx)
    for (let i=0;i<X.users.length;i++){
      const u = X.users[i], it = X.items[i];
      if (!map.has(u)) map.set(u, new Set());
      map.get(u).add(it);
    }
    return map;
  }

  // ---------- Wiring UI ----------
  async function parseFilesOrSample(useSample) {
    if (useSample) {
      state.rawRatings = DataIO.sample.ratings.slice();
      state.rawItems = DataIO.sample.items.slice();
    } else {
      const ratingsFile = $('ratingsFile').files[0];
      const itemsFile = $('itemsFile').files[0];
      if (!ratingsFile || !itemsFile) {
        alert('Please select both ratings (u.data) and movies (u.item) files.');
        return;
      }
      const [ratingsText, itemsText] = await Promise.all([
        readFileText(ratingsFile), readFileText(itemsFile)
      ]);
      state.rawRatings = DataIO.parseUData(ratingsText);
      state.rawItems = DataIO.parseUItem(itemsText);
    }

    // Build maps and tensors
    state.maps = DataIO.buildIdMaps(state.rawRatings);
    state.X = DataIO.applyIndexMaps(state.rawRatings, state.maps);
    state.titlesByIdx = DataIO.attachTitles(state.rawItems, state.maps.idx2ItemId);

    // Fill UI summaries & previews
    const nUsers = state.maps.idx2UserId.length;
    const nItems = state.maps.idx2ItemId.length;
    const nRatings = state.X.ratings.length;

    setSummary($('dataSummary'),
      `Users: <b>${nUsers}</b> · Movies: <b>${nItems}</b> · Ratings: <b>${nRatings}</b>`
    );

    const previewRows = state.rawRatings.slice(0, 20).map(r => [
      r.userId, r.movieId, (r.rating ?? '').toString()
    ]);
    setTable($('previewTable'), previewRows, ['userId', 'movieId', 'rating']);

    // Populate selects
    fillUserSelect();
    fillMovieSelect();

    // Model reset
    destroyModel();
    state.model = new MFRecommender(nUsers, nItems, Number($('k').value), Number($('reg').value));
    state.trained = false;
    setSummary($('trainStatus'), 'Model initialized. Ready to train.');

    // Model Anatomy summary
    setSummary($('embeddingSummary'),
      `Embedding Layers: <b>User</b> [${nUsers}×${$('k').value}] · <b>Item</b> [${nItems}×${$('k').value}]<br/>` +
      `Prediction: <code>p<sub>u</sub> · q<sub>i</sub> + b<sub>u</sub> + b<sub>i</sub> + μ</code>`
    );
    setTable($('latentUserTable'), [], []);
    setTable($('latentItemTable'), [], []);
    setTable($('predBreakdownTable'), [], []);
  }

  function fillUserSelect(){
    const sel = $('userSelect');
    sel.innerHTML = '';
    state.maps.idx2UserId.forEach((id, idx) => {
      const opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = `User ${id} (idx ${idx})`;
      sel.appendChild(opt);
    });
  }

  function fillMovieSelect(){
    const sel = $('movieSelect');
    sel.innerHTML = '';
    state.titlesByIdx.forEach((title, idx) => {
      const opt = document.createElement('option');
      opt.value = String(idx);
      opt.textContent = `${title} (idx ${idx})`;
      sel.appendChild(opt);
    });
  }

  function destroyModel(){
    if (!state.model) return;
    state.model.userEmb.dispose();
    state.model.itemEmb.dispose();
    state.model.userBias.dispose();
    state.model.itemBias.dispose();
    state.model.globalBias.dispose();
    state.model = null;
  }

  async function train() {
    if (!state.X) { alert('Please parse data first.'); return; }
    if (!state.model) {
      state.model = new MFRecommender(
        state.maps.idx2UserId.length,
        state.maps.idx2ItemId.length,
        Number($('k').value),
        Number($('reg').value)
      );
    }
    state.stopRequested = false;

    const params = {
      epochs: Number($('epochs').value),
      batchSize: Number($('batch').value),
      lr: Number($('lr').value),
      split: Number($('split').value)
    };

    // TFJS VIS: attach to container
    const container = { name: 'Training Curves', tab: 'Charts' };
    const history = { train: [], val: [] };

    await state.model.train(state.X, params, ({epoch, epochs, trainRmse, valRmse}) => {
      history.train.push({x: epoch, y: trainRmse});
      history.val.push({x: epoch, y: valRmse});
      tfvis.render.linechart(container, { values:[history.train, history.val], series:['Train RMSE','Val RMSE'] }, {
        xLabel:'Epoch', yLabel:'RMSE', width: 480, height: 300
      });
      setSummary($('trainStatus'),
        `Epoch ${epoch}/${epochs} — Train RMSE: <b>${prettyNum(trainRmse,3)}</b> · Val RMSE: <b>${prettyNum(valRmse,3)}</b>`
      );
    });

    if (!state.stopRequested) {
      state.trained = true;
      setSummary($('trainStatus'), $('trainStatus').innerHTML + '<br/><span style="color:var(--ok)">Training complete.</span>');
    } else {
      setSummary($('trainStatus'), $('trainStatus').innerHTML + '<br/><span style="color:var(--danger)">Training stopped.</span>');
    }
  }

  function stopTraining(){ state.stopRequested = true; }

  function recommend() {
    if (!state.model || !state.trained) { alert('Train the model first.'); return; }
    const userIdx = Number($('userSelect').value);
    const topN = Number($('topN').value);

    const ratedMap = buildRatedSet(state.X);
    const ratedSet = ratedMap.get(userIdx) || new Set();

    const recs = state.model.recommendForUser(userIdx, topN, ratedSet);
    // Render
    const rows = recs.map(r => {
      const title = state.titlesByIdx[r.itemIdx] || `Movie idx ${r.itemIdx}`;
      const rating = clamp(r.score, 1, 5);
      return [title, prettyNum(r.score,3), prettyNum(rating,2)];
    });
    setTable($('recsTable'), rows, ['Movie', 'Raw score', 'Clamped 1-5']);
  }

  function predictSingle() {
    if (!state.model || !state.trained) { alert('Train the model first.'); return; }
    const userIdx = Number($('userSelect').value);
    const itemIdx = Number($('movieSelect').value);
    const pred = state.model.predictSingle(userIdx, itemIdx);
    $('predOut').value = prettyNum(clamp(pred,1,5), 3) + ` (raw: ${prettyNum(pred,3)})`;
  }

  async function exportProcessed() {
    if (!state.X || !state.maps) { alert('Nothing to export yet.'); return; }
    const payload = {
      X: state.X,
      idx2UserId: state.maps.idx2UserId,
      idx2ItemId: state.maps.idx2ItemId,
      titlesByIdx: state.titlesByIdx
    };
    const blob = new Blob([JSON.stringify(payload)], {type:'application/json'});
    const filename = 'processed_movielens.json';
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    URL.revokeObjectURL(url);
    setSummary($('exportStatus'), `Exported processed data to <b>${filename}</b>.`);
  }

  async function importProcessed(ev) {
    const file = ev.target.files[0];
    if (!file) return;
    const text = await file.text();
    const obj = JSON.parse(text);
    state.X = obj.X;
    state.maps = {
      idx2UserId: obj.idx2UserId,
      idx2ItemId: obj.idx2ItemId,
      userId2Idx: new Map(obj.idx2UserId.map((v,i)=>[v,i])),
      itemId2Idx: new Map(obj.idx2ItemId.map((v,i)=>[v,i])),
    };
    state.titlesByIdx = obj.titlesByIdx || obj.idx2ItemId.map(id => `Movie ${id}`);
    fillUserSelect();
    fillMovieSelect();
    destroyModel();
    state.model = new MFRecommender(state.maps.idx2UserId.length, state.maps.idx2ItemId.length, Number($('k').value), Number($('reg').value));
    state.trained = false;
    setSummary($('dataSummary'),
      `Imported: Users <b>${state.maps.idx2UserId.length}</b> · Movies <b>${state.maps.idx2ItemId.length}</b> · Ratings <b>${state.X.ratings.length}</b>`
    );
    setSummary($('trainStatus'), 'Model initialized. Ready to train.');
    setSummary($('embeddingSummary'),
      `Embedding Layers: <b>User</b> [${state.maps.idx2UserId.length}×${$('k').value}] · <b>Item</b> [${state.maps.idx2ItemId.length}×${$('k').value}]<br/>` +
      `Prediction: <code>p<sub>u</sub> · q<sub>i</sub> + b<sub>u</sub> + b<sub>i</sub> + μ</code>`
    );
  }

  async function saveWeights() {
    if (!state.model) { alert('No model to save.'); return; }
    await state.model.save('mf');
  }

  async function loadWeights() {
    if (!state.model) { alert('Initialize model by parsing data first.'); return; }
    await state.model.load('mf');
    state.trained = true;
    setSummary($('trainStatus'), 'Weights loaded. Ready to recommend.');
  }

  // -------- Model Anatomy UI Actions --------
  function showUserVector(){
    if (!state.model) return alert('Initialize the model first.');
    const userIdx = Number($('userSelect').value);
    const vec = state.model.getUserVector(userIdx);
    const rows = [vec.map((v,i)=>`k${i}`), vec.map(v=>prettyNum(v,4))].map((row,i)=> (i===0? ['Dim', ...row] : ['Value', ...row]));
    // Convert to vertical table for readability
    const vertical = [];
    for (let i=0;i<vec.length;i++){
      vertical.push([`k${i}`, prettyNum(vec[i],4)]);
    }
    setTable($('latentUserTable'), vertical, ['User Latent Vector (p_u) dim', 'value']);
  }

  function showItemVector(){
    if (!state.model) return alert('Initialize the model first.');
    const itemIdx = Number($('movieSelect').value);
    const vec = state.model.getItemVector(itemIdx);
    const vertical = [];
    for (let i=0;i<vec.length;i++){
      vertical.push([`k${i}`, prettyNum(vec[i],4)]);
    }
    setTable($('latentItemTable'), vertical, ['Item Latent Vector (q_i) dim', 'value']);
  }

  function showPredictionBreakdown(){
    if (!state.model || !state.trained) return alert('Train the model first.');
    const userIdx = Number($('userSelect').value);
    const itemIdx = Number($('movieSelect').value);

    const pu = state.model.getUserVector(userIdx);
    const qi = state.model.getItemVector(itemIdx);
    const { ub, ib, mu } = state.model.getBiases(userIdx, itemIdx);

    // element-wise product and dot
    let dot = 0;
    const rows = [['k', 'p_u', 'q_i', 'p_u * q_i']];
    for (let i=0;i<pu.length;i++){
      const prod = pu[i] * qi[i];
      dot += prod;
      rows.push([`k${i}`, prettyNum(pu[i],4), prettyNum(qi[i],4), prettyNum(prod,4)]);
    }
    const predRaw = dot + ub + ib + mu;
    const predClamped = clamp(predRaw, 1, 5);

    // Add totals
    rows.push(['—','—','—','—']);
    rows.push(['Σ dot(p_u, q_i)', prettyNum(dot,4), '', '']);
    rows.push(['User bias (b_u)', prettyNum(ub,4), '', '']);
    rows.push(['Item bias (b_i)', prettyNum(ib,4), '', '']);
    rows.push(['Global mean (μ)', prettyNum(mu,4), '', '']);
    rows.push(['Prediction (raw)', prettyNum(predRaw,4), '', '']);
    rows.push(['Prediction (clamped 1–5)', prettyNum(predClamped,3), '', '']);

    setTable($('predBreakdownTable'), rows, ['component', 'value', '', '']);
    $('predOut').value = `${prettyNum(predClamped,3)} (raw: ${prettyNum(predRaw,4)})`;
  }

  // ---------- Event bindings ----------
  $('parseBtn').addEventListener('click', () => parseFilesOrSample(false));
  $('loadSample').addEventListener('click', () => parseFilesOrSample(true));
  $('trainBtn').addEventListener('click', train);
  $('stopBtn').addEventListener('click', stopTraining);
  $('recommendBtn').addEventListener('click', recommend);
  $('predictPairBtn').addEventListener('click', predictSingle);
  $('exportJsonBtn').addEventListener('click', exportProcessed);
  $('importJsonInput').addEventListener('change', importProcessed);
  $('saveBtn').addEventListener('click', saveWeights);
  $('loadWeightsBtn').addEventListener('click', loadWeights);

  // Model Anatomy buttons
  $('showUserVecBtn').addEventListener('click', showUserVector);
  $('showItemVecBtn').addEventListener('click', showItemVector);
  $('showBreakdownBtn').addEventListener('click', showPredictionBreakdown);

  // Friendly defaults
  setSummary($('dataSummary'), 'Load your files or click “Load tiny sample”.');
  setSummary($('trainStatus'), 'Waiting for data...');
  setSummary($('embeddingSummary'), 'Embeddings are initialized after you parse data.');
})();
