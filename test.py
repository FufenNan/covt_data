
# for i, item in enumerate(list_data_dict[:3]):
#     print(f"\n===== sample {i} =====")
#     print(item)

# {'id': 'identity_177477', 'image': 'identity_177477.png', 
#  'conversations': [{'from': 'human', 'value': '<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: If each friend spent the same amount of time at soccer practice, who left the practice last?'}, 
#                    {'from': 'gpt', 'value': 'The answer is Sam'}]}

from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("/u/yli8/jiaqi/CoVT-Dataset/images/anchor_outputs/000000000077_sam.png")

plt.imshow(img)
plt.axis("off")

scp username@server:/u/yli8/jiaqi/CoVT-Dataset/images/anchor_outputs/000000000077_sam.png .