from src.pipeline import run_pipeline

df = run_pipeline('phenotype clustering latent class')
print('Records found:', len(df))
print('Columns:', df.columns.tolist())
if len(df) > 0:
    print(df.head())