import cohere
co = cohere.Client('U5ORFB9ZcZOFu92c0U8RjVFtnEIqM3Xt1v2hZZsT') # This is your trial API key
response = co.generate(
  model='command',
  prompt='tell me a joke\n',
  max_tokens=300,
  temperature=0.9,
  k=0,
  stop_sequences=[],
  return_likelihoods='NONE')
print('Prediction: {}'.format(response.generations[0].text))