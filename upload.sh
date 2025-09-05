
echo '{"title": "MUSDB18-HQ Apollo Pre Process", "id": "adigen/musdb18-hq-apollo-preprocess", "licenses": [{"name": "CC0-1.0"}]}' > /tmp/prism/Apollo-Upgrade/hdf5_datas_new/dataset-metadata.json
cd /tmp/prism/Apollo-Upgrade
kaggle datasets version -p hdf5_datas --dir-mode=tar -m idkman