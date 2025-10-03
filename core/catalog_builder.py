import json, re
classes = sorted(df["ata04"].unique())


vec_all = TfidfVectorizer(min_df=2, ngram_range=(1,2), max_features=50000)
X_all = vec_all.fit_transform(df["text_norm"].tolist())


idx_by_cls = {ata: list(df.index[df["ata04"]==ata].values) for ata in classes}


title_map = {}
if Path(ATA_PARQUET).exists():
amap = pd.read_parquet(ATA_PARQUET)
for _, r in amap.iterrows():
title_map[str(r["ATA04"])] = str(r.get("Title","") or "")


inv_vocab = {v:k for k,v in vec_all.vocabulary_.items()}
def top_terms(row_ids):
sub = X_all[row_ids]
mean_vec = np.asarray(sub.mean(axis=0)).ravel()
top_idx = mean_vec.argsort()[::-1][:top_k]
return [inv_vocab[i] for i in top_idx if i in inv_vocab]


def rep_samples(row_ids):
sub = X_all[row_ids]
scores = np.asarray(sub.sum(axis=1)).ravel()
ranked = [row_ids[i] for i in np.argsort(scores)[::-1]]
out = []
for ridx in ranked:
t = str(df.loc[ridx, "text"]).strip()
if len(t) < 40: continue
out.append(t[:240])
if len(out) >= sample_k: break
if len(out) < sample_k:
for ridx in ranked[:sample_k-len(out)]:
out.append(str(df.loc[ridx,"text"])[:240])
return out[:sample_k]


catalog = {}
ata_list = []
for ata in classes:
row_ids = idx_by_cls[ata]
if len(row_ids) < min_docs_per_class:
kws = top_terms(row_ids) if row_ids else []
sps = rep_samples(row_ids) if row_ids else []
else:
kws = top_terms(row_ids)
sps = rep_samples(row_ids)
title = title_map.get(ata, "")
catalog[ata] = {"title": title, "keywords": kws, "samples": sps}
ata_list.append(ata)


docs = []
for ata in ata_list:
info = catalog[ata]
doc = " ".join([info.get("title",""), " ".join(info.get("keywords",[])), " ".join(info.get("samples",[]))]).strip()
docs.append(doc if doc else ata)


vec_cat = TfidfVectorizer(min_df=1, ngram_range=(1,2))
X_cat = vec_cat.fit_transform(docs)


Path(OUT_JSON).write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
dump(vec_cat, OUT_VEC)
save_npz(OUT_MAT, X_cat)


stat = pd.DataFrame({"ATA04": ata_list, "Docs": [len(idx_by_cls[a]) for a in ata_list]})
return stat.sort_values(["Docs","ATA04"], ascending=[False,True]).reset_index(drop=True)
