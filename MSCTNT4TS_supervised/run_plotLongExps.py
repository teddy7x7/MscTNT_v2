import numpy as np
import os
from utils.tools import visual_multiPlot


# 要用來切分第一個維度，將矩陣還原成 [batch_num, batch_size, seq_len, channels]時使用
batch_size = 128

exp_plot_name = "weather_FIN_trySmaller3_diffPatchLen3_720_720_plot02_feature-2"
plot_folder_path = "test_results/" + exp_plot_name

# 每個元素儲存要放在plot上某一個參數的實驗結果對應的矩陣
predFilePath_list=[]

# 要讀取的預測結果矩陣們
# predFilePath_1 = "results/national_illness_forPlot2_120_24_MscTNT4TS_custom_ftM_sl120_ll48_pl24_dm512_nh8_el1_dl1_df2048_fc1_ebtimeF_dtTrue_'Exp'_0/pred.npy"
predFilePath_1 = "results/weather_FIN_trySmaller3_diffPatchLen3_720_720_pthLen12_spthLen1_MscTNT4TS_custom_ftM_sl720_ll48_pl720_dm512_nh8_el1_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/pred.npy"
predFilePath_list.append(predFilePath_1)

predFilePath_2 = "results/weather_FIN_trySmaller3_diffPatchLen3_720_720_pthLen60_spthLen20_MscTNT4TS_custom_ftM_sl720_ll48_pl720_dm512_nh8_el1_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/pred.npy"
predFilePath_list.append(predFilePath_2)

predFilePath_3 = "results/weather_FIN_trySmaller3_diffPatchLen3_720_720_pthLen120_spthLen40_MscTNT4TS_custom_ftM_sl720_ll48_pl720_dm512_nh8_el1_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/pred.npy"
predFilePath_list.append(predFilePath_3)

predFilePath_4 = "results/weather_FIN_trySmaller3_diffPatchLen3_720_720_pthLen360_spthLen180_MscTNT4TS_custom_ftM_sl720_ll48_pl720_dm512_nh8_el1_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/pred.npy"
predFilePath_list.append(predFilePath_4)

patchTST_predFilePath = "results/weather_PatchTST64_50epoch_512_720_PatchTST_custom_ftM_sl512_ll48_pl720_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_Exp_0/pred.npy"
predFilePath_list.append(patchTST_predFilePath)

predFile_npy_list=[]

for i, predFilePath in enumerate(predFilePath_list):
    print(f"load pred_{i}'s pred.npy from :\n", predFilePath)
    npArray = np.load(file=predFilePath)
    print(f"shape of pred_{i} before reshape: {npArray.shape}")
    npArray=npArray.reshape(-1, batch_size, npArray.shape[-2], npArray.shape[-1])
    print(f"shape of pred_{i} after reshape: {npArray.shape}")
    predFile_npy_list.append(npArray)
    print("After append, len(predFile_npy_list) : ", len(predFile_npy_list))
      

# 要讀取的ground_truth矩陣
trueFilePath = "results/weather_FIN_trySmaller3_diffPatchLen3_720_720_pthLen12_spthLen1_MscTNT4TS_custom_ftM_sl720_ll48_pl720_dm512_nh8_el1_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/true.npy"
# trueFilePath = None

# print("load pred_01's pred.npy from :\n", predFilePath_1)
# preds_01 = np.load(file=predFilePath_1)

if trueFilePath is not None:
    print("load true.npy from :\n", trueFilePath)
    trues = np.load(file=trueFilePath)
    # shape of true array : (160, 24, 7), ie. (batch_size*batch_num, seq_len, channels)
    print("shape of true array :", trues.shape)
    trues = trues.reshape(-1, batch_size,trues.shape[-2], trues.shape[-1]) # [batch_num, batch_size, seq_len, channels]
    print("shape of true array after reshape :", trues.shape)

# shape of pred array : (160, 24, 7), ie. (batch_size*batch_num, seq_len, channels)
# print("shape of pred_01 array :", preds_01.shape)

# preds_01 = preds_01.reshape(-1, batch_size, preds_01.shape[-2], preds_01.shape[-1])
# print("shape of pred array after reshape :", preds_01.shape)

if not os.path.exists(plot_folder_path):
            os.makedirs(plot_folder_path)

# for i in range(trues.shape[0]):
#     if i%20==0:
#         pred_01_batch = preds_01[i]
#         true_batch = trues[i]
#         print("pred_01_batch.shape : ", pred_01_batch.shape)
#         print("true_batch.shape : ", true_batch.shape)

#         print("pred_01_batch[0][:][-1].shape : ", pred_01_batch[0][:][-1].shape)

        # 取batch中的第一個sequence，全部長度，最後一個channel來畫圖
        # visual(true_batch[0][:][-1], pred_01_batch[0][:][-1], os.path.join(plot_folder_path, str(i) + '.pdf'))

print("batch_nums(ie. predFile_npy_list[0].shape[0].shape ) : ", predFile_npy_list[0].shape[0])
print("diff_results_nums(ie. len(predFile_npy_list) ) : ", len(predFile_npy_list))


for i in range(predFile_npy_list[0].shape[0]):

    # 每20個batch
    if i%20==0:
        # for j, result in enumerate(predFile_npy_list):
        #     print(f"{j}th result's {i}th batch")
        #     print(f"result.shape : ",result.shape)
        #     print(f"result[{i}].shape : ",result[i].shape)
        #     print(f"result[{i}, 0].shape : ",result[i, 0].shape)
        #     print(f"result[{i}, 0, :].shape : ",result[i, 0, :].shape)
        #     print(f"result[{i}, 0, :, -1].shape : ",result[i, 0, :, -1].shape)
        #     print("\n")
        # diff_results_list = [result[i,0,:,-1] for result in predFile_npy_list]
        diff_results_list = [result[i,0,:,-2:-1] for result in predFile_npy_list]
        # print("diff_results_list.shape : ", np.array(diff_results_list).shape)
        
        true_batch = trues[i] # 第i個batch
        # true_seq = true_batch[0, :, -1]
        true_seq = true_batch[0, :, -2:-1]
        print("true_batch.shape : ", true_batch.shape)
        print(f"true_seq.shape : ", true_seq.shape)
        # true_seq = None

        visual_multiPlot(true_seq, diff_results_list, os.path.join(plot_folder_path, str(i) + '.pdf'))



