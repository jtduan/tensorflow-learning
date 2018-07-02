# coding=utf-8
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

arr = []
arr_inner = []
for index in range(5):
    # 读取灰度图
    img_src = cv2.imread("./temp/anjuke" + str(index) + ".jpg", cv2.COLOR_RGB2GRAY)
    b, g, r = cv2.split(img_src)
    img = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)
    # 获取轮廓图
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, 3)

    # 读取模板图
    template = cv2.imread("./temp/ajk-logo.jpg", cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)

    print maxVal
    cv2.rectangle(img_src, (maxLoc[0], maxLoc[1]), (maxLoc[0] + w, maxLoc[1] + h),
                  (0, 255, 0), 3)

    _, img_bin = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY)

    for j in range(w):
        min = 10000
        max = 0
        for i in range(h):
            if img_bin[i][j] == 255:
                min = i if min > i else min
                max = i if max < i else max
        if (min < 1000):
            before = after = min
            while img_bin[before][j] == 255:
                before = before - 1
            while img_bin[after][j] == 255:
                after = after + 1
            # print before, after
            # img_src[maxLoc[1] + before][maxLoc[0] + j][:] = 0
            # img_src[maxLoc[1] + after][maxLoc[0] + j][:] = 66

            # cv2.imshow('dst', img_src)
            # cv2.waitKey(0)
            # cv2.destroyAllWindow()
            arr.append(b[maxLoc[1] + before][maxLoc[0] + j])
            arr_inner.append(b[maxLoc[1] + after][maxLoc[0] + j])

        # if (max > 0):
        #     before = after = max
        #     while img_bin[before][j] == 255:
        #         before = before + 1
        #     while img_bin[after][j] == 255:
        #         after = after - 1
        #     # print before, after
        #     # img_src[maxLoc[1] + before][maxLoc[0] + j][:] = 0
        #     # img_src[maxLoc[1] + after][maxLoc[0] + j][:] = 66
        #
        #     # cv2.imshow('dst', img_src)
        #     # cv2.waitKey(0)
        #     # cv2.destroyAllWindow()
        #     arr.append(b[maxLoc[1] + before][maxLoc[0] + j])
        #     arr_inner.append(b[maxLoc[1] + after][maxLoc[0] + j])

plt.scatter(arr, arr_inner,alpha=0.5)
plt.show()

submission = pd.DataFrame(data={'before': arr, 'after': arr_inner})
submission.to_csv('submission.csv', index=False)
