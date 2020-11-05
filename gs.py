
import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
import detect
import tensorflow as tf
import time

#Curently not working
def draw_objects(draw, obj, labels,height,width):
  input_image = np.asarray(Image.fromarray(np.uint8(draw)))
  image = input_image[:, :, ::-1].copy()
  b = np.array(obj).astype(int)
  cv2.rectangle(image, (b[1], b[0]), (b[3], b[2]), (0,0,255), 2, cv2.LINE_AA)
  print("bounding box : {}  {}  {}  {}".format(b[0], b[1], b[2], b[3]))
  """Draws the bounding box and label for each object."""
  # draw = draw.text((xmin + 10, ymin + 10),'%s\n%.2f' % (labels[2],60),fill='red')
  # draw.save("edit.jpg")
  im = Image.fromarray(image)
  im.save("./d/ab.jpg")
  # draw.show()



def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def  load_model(model_file):
  interpreter = tf.lite.Interpreter(model_path=model_file)
  interpreter.allocate_tensors()
  return interpreter

def  detect_classes(interpreter,image,input_details,output_details,img):
  image2 = image.copy()
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  image = image.resize((width, height))


  # add N dim
  input_data = np.expand_dims(image, axis=0)

  scale = detect.set_input(interpreter, image.size,lambda size: image.resize(size, Image.ANTIALIAS))

  interpreter.invoke()

  tensor = interpreter.tensor(interpreter.get_output_details()[3]['index'])()
  count = int(np.squeeze(tensor))
  
  tensor = interpreter.tensor(interpreter.get_output_details()[1]['index'])()
  class_id = np.squeeze(tensor)

  tensor = interpreter.tensor(interpreter.get_output_details()[0]['index'])()
  box = np.squeeze(tensor)

  tensor = interpreter.tensor(interpreter.get_output_details()[2]['index'])()
  Probability = np.squeeze(tensor)

  # print(interpreter.tensor(16)())
  labels = load_labels(label_file)
  _, height, width, _ = interpreter.get_input_details()[0]['shape']
  image = image.convert('RGB')
  retstr = []
  net_w,net_h = 300,300
  
  x_offset, x_scale = (net_w - net_w)/2./net_w, float(net_w)/net_w
  y_offset, y_scale = (net_h - net_h)/2./net_h, float(net_h)/net_h

  for i in range(count):
    if Probability[i]>0.50:
      box[i][0] = int((box[i][0] - x_offset) / x_scale * width)
      box[i][2] = int((box[i][2] - x_offset) / x_scale * width)
      box[i][1] = int((box[i][1] - y_offset) / y_scale * height)
      box[i][3] = int((box[i][3] - y_offset) / y_scale * height)

      retstr.append("Class : {} , Probability : {}.".format(labels[int(class_id[i])],Probability[i]))
      print("Class : {} , Probability : {} , Bounding_box : {} ".format(labels[int(class_id[i])],Probability[i],box[i]))
      # draw_objects(image, box[i], labels,height,width)
  # image.save("image.jpg")
  # cv2.waitKey(0)

	
if __name__ == '__main__':


  model_file = "detect.tflite"
  label_file = 'labelmap.txt'

  interpreter = load_model(model_file)

  # interpreter = tf.lite.Interpreter(model_path=model_file)
  # interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # cap = cv2.VideoCapture(0)
  
  while True:
    start_time = time.time()
    # cap = cv2.VideoCapture(0)
    # _,img = cap.read()
    img = cv2.imread("n.jfif")
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img2)
    detect_classes(interpreter,im_pil,input_details,output_details,img)
    print("--- %s seconds ---" % (time.time() - start_time))
    break
  # cap.release()
    # time.sleep(5)