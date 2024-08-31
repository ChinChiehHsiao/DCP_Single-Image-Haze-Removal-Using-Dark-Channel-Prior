# (2011) DCP_Single Image Haze Removal Using Dark Channel Prior
![compare](https://github.com/user-attachments/assets/4b746475-9951-48d8-ac08-85ff9cfa873a)<br><br>
![compare 2](https://github.com/user-attachments/assets/a7f97f8d-5d0d-4171-9873-8b66655423c9)


References : HE, Kaiming; SUN, Jian; TANG, Xiaoou. Single image haze removal using dark channel prior. IEEE transactions on pattern analysis and machine intelligence, 2010, 33.12: 2341-2353.<br><br>
暗通道先驗圖像去霧文章復現<br>
使用python<br><br>
- 原文PDF &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;：[點擊此連結](https://mmlab.ie.cuhk.edu.hk/archive/2011/Haze.pdf)
- 原文DOI &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;：[點擊此連結](https://doi.org/10.1109/TPAMI.2010.168)<br><br>


操作說明 
---
把圖片放進 `my dataset` 資料夾，打開 `DCP.py` 檔案執行即可<br>
結果圖將儲存在 `my result` 資料夾中。<br><br>


result檔案名稱說明
---
- 原圖名稱
- DCP_dehaze &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; : 最終去霧結果圖
- All detail image &nbsp; : 依序為  原圖, 原圖暗通道, 原圖/A, (原圖/A)暗通道, ((原圖/A)暗通道)local max, 初估的transmission, 使用guided filter refine的transmission, 最終去霧結果圖<br><br>

---
關於
---

- 復現 : 蕭晉杰
- 時間 : 2024.08.27
