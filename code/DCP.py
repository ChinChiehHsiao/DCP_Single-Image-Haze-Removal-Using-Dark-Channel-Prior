import numpy as np
import os
import cv2
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.font_manager import fontManager
#for i in sorted(fontManager.get_font_names()):
#    print(i)
matplotlib.rc('font', family='Microsoft JhengHei')



def plot__BGR(h, w, num, image, title):
    plt.subplot(h, w, num)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")

def plot_gray(h, w, num, image, title):
    plt.subplot(h, w, num)
    plt.imshow(image, cmap = 'gray', vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")



def dark_channel(image) : 
    H, W, _ = image.shape
    patch_size = 15
    pad_size = patch_size // 2
    #創建 H*W 全零矩陣
    dc = np.zeros((H, W), dtype=np.float32)
    #填充邊界
    #用無限大的值填充的用意是取local min才不會影響到取值
    imJ = np.pad(image ,((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='constant', constant_values=np.inf)   
    #計算暗通道
    for j in range(H):
        for i in range(W):
            #遍例3通道, 所有patch為15
            patch = imJ[j:(j+patch_size), i:(i+patch_size),:]
            #將patch中抓到的local min存到dc(j, i)裡
            dc[j, i] = np.min(patch)
            
    return dc



def local_maximum(image):
    # 單通道使用
    H, W = image.shape
    patch_size = 15
    pad_size = patch_size // 2
    # 創建 H*W 全零矩陣
    local_max = np.zeros((H, W), dtype=np.float32)
    # 填充邊界
    # 用無限大的值填充的用意是取local max才不會影響到取值
    imJ = np.pad(image ,((pad_size, pad_size), (pad_size, pad_size)), mode='constant', constant_values=-np.inf)
    # 計算local max
    for j in range(H):
        for i in range(W):
            patch = imJ[j:(j+patch_size), i:(i+patch_size)]
            local_max[j, i] = np.max(patch)
            
    return local_max



def atmospheric_light(image, dark_channel):
	#在暗通道中找最亮的像素，並在原彩圖中找到對應的位置，來計算大氣光的顏色

	H, W, _ = image.shape
	imsize = H * W													#計算圖像總像素數量
	numpx = np.floor(imsize / 1000).astype(int)						#計算要選擇的像素數量，選前0.1%，並向下取整
	dark_channel_Vec = dark_channel.ravel()							#將dark_channel展平成 imsize *1 的列向量
	ImVec = image.reshape(imsize, 3)								#將原始影像image展平成 imsize * 3 的矩陣
	indices = np.argsort(dark_channel_Vec)							#對暗通道進行升序排序，indices=索引值
	indices = indices[-numpx:]										#選擇排序後最亮的0.1%，計算從哪裡開始提取索引的起點，並到end
	atmSum = np.zeros(3, dtype=np.float32)							#計算大氣光的顏色，創建 1 * 3 全0矩陣，用來儲存最亮像素的顏色值累加和

	#遍例所有選中的像素
	for ind in range(numpx):
		atmSum += ImVec[indices[ind]]								#將每個選中像素的RGB顏色值累加到atmSum

	A = atmSum / numpx												#將atmSum取平均，即為大氣光的顏色

	return A



def main_dehaze(image_name):

    image = (cv2.imread(image_name).astype(np.float32))/255

    H, W, C = image.shape
    I_dark = dark_channel(image)
    A = atmospheric_light(image, I_dark)
    image_A = np.zeros_like(image)
    for i in range(C):
        image_A[:, :, i] = image[:, :, i]/A[i]
    image_A_dark = dark_channel(image_A)
    image_A_dark_local_max = local_maximum(image_A_dark)
    t = 1 - 0.95 * image_A_dark_local_max

    radius = round(np.minimum(H, W) / 50)         # https://kaiminghe.github.io/publications/thesis.pdf   p.95
    # radius = 60
    eps = 1e-4
    I_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    GF_t = cv2.ximgproc.guidedFilter(guide=I_gray, src=t, radius=radius, eps=eps)

    t0 = 0.1
    I_dehaze = np.zeros_like(image)
    for i in range(C):
        I_dehaze[:, :, i] = (image[:, :, i] - A[i])/cv2.max(GF_t, t0) + A[i]
    I_dehaze = np.clip(I_dehaze, 0, 1)

    return image, I_dark, image_A, image_A_dark, image_A_dark_local_max, t, GF_t, I_dehaze



input_folder  = "my dataset"
output_folder = "my result"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_count = 0

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        image_count += 1
        image_path = os.path.join(input_folder, filename)
        file_base_name = os.path.splitext(filename)[0]
        print(file_base_name)

        individual_output_folder = os.path.join(output_folder, f"{file_base_name}")
        if not os.path.exists(individual_output_folder):
            os.makedirs(individual_output_folder)

        image, I_dark, image_A, image_A_dark, image_A_dark_local_max, t, GF_t, I_dehaze = main_dehaze(image_path)

        output_path_ori = os.path.join(individual_output_folder, f"{file_base_name}.png")
        cv2.imwrite(output_path_ori, (image * 255).astype(np.uint8))
        output_path_I_dehaze = os.path.join(individual_output_folder, f"DCP_dehaze ({file_base_name}).png")
        cv2.imwrite(output_path_I_dehaze, (I_dehaze * 255).astype(np.uint8))

        plt.figure(figsize=(22, 15))
        h, w = 1, 8
        plot__BGR(h, w, 1, image,                   "original image"                 )
        plot_gray(h, w, 2, I_dark,                  "I_dark"                         )
        plot__BGR(h, w, 3, np.clip(image_A, 0, 1),  "image/A"                        )
        plot_gray(h, w, 4, image_A_dark,            "( image/A )_dark"               )
        plot_gray(h, w, 5, image_A_dark_local_max,  "( ( image/A )_dark )_local max" )
        plot_gray(h, w, 6, t,                       "t"                              )
        plot_gray(h, w, 7, np.clip(GF_t, 0, 1),     "GF_t"                           )
        plot__BGR(h, w, 8, I_dehaze,                "DCP result"                     )
        plt.tight_layout()
        output_path_All_detail_image = os.path.join(individual_output_folder, f"All detail image ({file_base_name}).png")
        plt.savefig(output_path_All_detail_image, dpi=300, bbox_inches='tight')
        # plt.show()
plt.show()
# plt.close('all')

print("===========================================================")
# print("\n")
print(f"總共讀入了 {image_count} 張圖片")