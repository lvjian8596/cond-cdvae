
# carbon edip 10 GPa

PROJECT_DIR=${HOME}/cond_cdvae
model_path=${PROJECT_DIR}/cond-carbon-edip-10GPa/with_comp-t14/
tgt_key=energy_per_atom
tgt_arg=--${tgt_key}
tgt_val=-1

if [ ! -e ${model_path}/eval_recon.pt ]; then
python -u ~/cond-cdvae/scripts/evaluate.py --tasks recon \
    --model_path ${model_path}
fi

# python ~/cond-cdvae/scripts/evaluate.py --tasks gen \
#     --model_path ${model_path} \
#     --batch_size 100 \
#     --${tgt_key} ${tgt_val}
# mkdir -p ${model_path}/gen_${tgt_val}
# mv ${model_path}/eval_gen.pt ${model_path}/gen_${tgt_val}
# python ~/cond-cdvae/scripts/extract_gen.py ${model_path}/gen_${tgt_val}
# python ~/cond-cdvae/scripts/valid_c.py ${model_path}/gen_${tgt_val}/eval_gen.pt
# python ~/cond-cdvae/scripts/gulp/carbon2gulpin.py -p 10 10 -j 60 valid_c/*.vasp
# python ~/cond-cdvae/scripts/gulp/batch_gulp.py
# python ~/cond-cdvae/scripts/gulp/read_gulp.py
