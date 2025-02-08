import cv2
import matplotlib.pyplot as plt
from traffic_sign import process_image
from lane_detection import lane_finding_pipeline

# Input image path
image_path = 'E:/project/Computer vision sem5/test_images/d3.jpeg'

# Task 1: Classification Results
s_class, s_meaning, sign, i_class, i_meaning, enhanced_sign, predicted_class, sign_meaning = process_image(image_path)
canny_edges, cropped_image, final = lane_finding_pipeline(image_path)

# Display Input Image
plt.figure(figsize=(20, 20))
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))
plt.title(f"Input Image -{s_class}, Meaning: {s_meaning}")
plt.axis('off')

plt.subplot(2, 3, 2)
if sign is not None and sign.size > 0:
  plt.imshow(cv2.cvtColor(sign, cv2.COLOR_BGR2RGB))
  plt.title(f"Sign - {i_class}, Meaning: {i_meaning}")
else:
  plt.title("No sign detected")
plt.axis('off')

plt.subplot(2, 3, 3)
if enhanced_sign is not None and sign.size > 0:
   plt.imshow(cv2.cvtColor(enhanced_sign, cv2.COLOR_BGR2RGB))
   plt.title(f"Enhanced Sign - {predicted_class}, Meaning: {sign_meaning}")
else:
   plt.title("No sign detected")
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(canny_edges, cv2.COLOR_BGR2RGB))
plt.title("Canny Image")
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
plt.title("Region of intrest")
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.title("Final image")
plt.axis('off')

plt.show()
