#!pip install --quiet gradio
#import gradio as gr 

# def only_numerics(seq):
#   seq_type= type(seq)
#   return seq_type().join(filter(seq_type.isdigit, seq))

# def age(model):
#   k = only_numerics(model)
#   k = int(k)
#   assert k < 15 and k > 5, f'We dont have the data for Iphone {k}'
#   age = 14-k
#   return age


# Convert the other ifs to elifs

def depreciation(model,AGE, percentage_crack):
  print(model,model == "iphone 7")
  #model = model.strip().lower()
  if percentage_crack < 1:
    
    CDPR = 0.35 * percentage_crack

  if model == "iphone 7" or model == 'Iphone 7':
    RCV=1000
  
  if model == 'iphone 8' or model == 'Iphone 8' :
    RCV=1000
  
  if model == 'iphone 10' or model == 'Iphone 10':
    RCV=1000
  
  if model == 'iphone 11' or model == 'Iphone 11':
    RCV=1000
  
  if model == 'iphone 12' or model == 'Iphone 12':
    RCV=1000
  
  if model == 'iphone 13' or model == 'Iphone 13':
    RCV=1000
  
  if model == 'iphone 14' or model == 'Iphone 14':
    RCV=1000
  
  else:
    return ('There is no data for your model in our database')

  DPR = 0.25 + 0.2 * AGE
  ACV = RCV - DPR * RCV * AGE * CDPR
  
  return ACV


# image = gr.inputs.Image(shape=(32, 32))

# demo = gr.Interface(
#     fn=depreciation,
#     inputs=[image, 'text', 'number'],
#     outputs=['number'], 
#     title = "Depreciation Estimation of Phone",
#     description = "Input necessary features for price estimation"
# )

# Launch the interface
# demo.launch(debug=True)