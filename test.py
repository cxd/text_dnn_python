from lib import GloveReader

reader = GloveReader.GloveReader(base_dir='data')

print(reader.model_sizes.keys)

model1 = reader.read_glove_model('model50')