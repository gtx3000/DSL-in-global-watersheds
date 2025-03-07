import numpy as np
import pandas as pd
import math
import scipy.signal as signal
import os
import shutil


def Harmonics(coefa, coefb, hvar, tseries, nmodes):
    mtot = len(tseries)  # retrieving the length of the time dimension
    time = np.arange(1, mtot + 1, 1.)  # Just a an array of increasing numbers
    tdata = tseries[:]  # Adjusting the mean annual cycle

    svar = sum((tdata[:] - np.mean(tdata)) ** 2) / (mtot - 1)
    nm = nmodes
    if 2 * nm > mtot:
        nm = mtot / 2
    coefa = np.zeros((nm))
    coefb = np.zeros((nm))
    hvar = np.zeros((nm))
    for tt in range(0, nm):
        Ak = np.sum(tdata[:] * np.cos(2. * math.pi * (tt + 1) * time[:] / float(mtot)))
        Bk = np.sum(tdata[:] * np.sin(2. * math.pi * (tt + 1) * time[:] / float(mtot)))
        coefa[tt] = Ak * 2. / float(mtot)
        coefb[tt] = Bk * 2. / float(mtot)
        hvar[tt] = mtot * (coefa[tt] ** 2 + coefb[tt] ** 2) / (2. * (mtot - 1) * svar)
    return coefa, coefb, hvar


if __name__=="__main__":
    #流域p-e 去趋势
    data_path=r"D:\code\data\basin\hydrobasins"
    shp_path=r"D:\code\data\basin\hydrobasins\basin_shp"
    desert=r"D:\code\data\basin\hydrobasins\basin_desert100"

    p_path=r"D:\code\data\basin\hydrobasins\p_basin"
    e_path=r"D:\code\data\basin\hydrobasins\e_basin"
    files=os.listdir(e_path)
    files.remove("e_fig")

    dso_list = np.zeros((len(files), 40))
    dse_list = np.zeros((len(files), 40))
    dsl_list = np.zeros((len(files), 40))
    dso_mean_list=np.zeros((len(files),2))
    dan_name = np.zeros((len(files)), dtype=object)
    num = 0

    for file in files:
        if(os.path.isfile(desert+"\\"+file[:-4]+".shp")):
            continue

        precip_basin=np.load(p_path+"\\"+file)
        evapor_basin=np.load(e_path+"\\"+file)

        # 滑动平均，保证平均前后序列长度一样
        p_41_30run=np.zeros(15005)
        p_41_30run[:15]=precip_basin[:15]
        p_41_30run[-14:]=precip_basin[-14:]
        p_41_30run[15:-14]=precip_basin[:]

        e_41_30run=np.zeros(15005)
        e_41_30run[:15]=evapor_basin[:15]
        e_41_30run[-14:]=evapor_basin[-14:]
        e_41_30run[15:-14]=evapor_basin[:]

        window=np.repeat(1/30,30)
        p_41_smooth=np.convolve(p_41_30run,window,mode="valid")
        e_41_smooth=np.convolve(e_41_30run,window,mode="valid")

        p_e_smooth=np.zeros(14976)
        for i in range(14976):
            p_e_smooth[i]=p_41_smooth[i]-e_41_smooth[i]

        # 计算41年日历年的每日均值
        p_e_41 = np.zeros((41, 366))
        start = 0
        for yr in range(41):
            if (yr + 1980) % 4 == 0:
                end = 366
            else:
                end = 365
            p_e_41[yr, 0:end] = p_e_smooth[start:start + end]
            start = start + end

        p_e_41 = np.delete(p_e_41, -1, axis=1)  # 有些年份第366项是0，所以去除
        p_e_41_mean = np.mean(p_e_41, axis=0)  # 行变为1行，按列求平均，得到41年每日均值
        p_e_mean=np.mean(p_e_41_mean)

        p_e_ana_mean = np.zeros(365)
        for i in range(365):
            p_e_ana_mean[i] = np.sum(p_e_41_mean[0:i + 1]) - (i + 1) * p_e_mean


        coefa = np.zeros((3))
        coefb = np.zeros((3))
        hvar = np.zeros((3))
        tseries = p_e_ana_mean[:]
        coefa, coefb, hvar = Harmonics(coefa, coefb, hvar, tseries, 3)

        #删除一年两季流域
        if hvar[1] > hvar[0]:
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".dbf", data_path + "\\" + "e_two_seasons" + "\\" + file[:-4] + ".dbf")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".prj", data_path + "\\" + "e_two_seasons" + "\\" + file[:-4] + ".prj")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".shp", data_path + "\\" + "e_two_seasons" + "\\" + file[:-4] + ".shp")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".shx", data_path + "\\" + "e_two_seasons" + "\\" + file[:-4] + ".shx")
            continue

        # 寻找每年大概的旱季时间
        max_index=signal.argrelmax(p_e_ana_mean)
        if len(max_index[0])==0:
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".dbf", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".dbf")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".prj", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".prj")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".shp", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".shp")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".shx", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".shx")
            continue
        else:
            dso_mean=max_index[0][0]
            for i in range(len(max_index[0])):
                if p_e_ana_mean[max_index[0][i]]>p_e_ana_mean[dso_mean]:
                    dso_mean=max_index[0][i]

        min_index=signal.argrelmin(p_e_ana_mean)
        if len(min_index[0])==0:
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".dbf", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".dbf")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".prj", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".prj")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".shp", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".shp")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".shx", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".shx")
            continue
        else:
            dse_mean=min_index[0][0]
            for i in range(len(min_index[0])):
                if p_e_ana_mean[min_index[0][i]]<p_e_ana_mean[dse_mean]:
                    dse_mean=min_index[0][i]

        dsl_mean = dse_mean - dso_mean
        if dsl_mean < 0:
            dsl_mean = 365 + dsl_mean

        if dsl_mean<=20:
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".dbf", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".dbf")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".prj", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".prj")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".shp", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".shp")
            shutil.copy(data_path + "\\basin_shp\\" + file[:-4] + ".shx", data_path + "\\" + "e_humid" + "\\" + file[:-4] + ".shx")
            continue


         # 用 水文年 分割
        precip_year = np.zeros((40, 366))
        evapor_year=np.zeros((40, 366))
        dso_mean = dso_mean - 60
        if dso_mean < 0:  # 假设dso最大波动为向前60天，防止出现负数
            dso_mean = 365 + dso_mean
        start = dso_mean

        for yr in range(40):
            if (yr + 1980) % 4 == 0:
                end = 366
            else:
                end = 365
            precip_year[yr, 0:end] = p_e_smooth[start:start + end]
            start = start + end
        precip_year = np.delete(precip_year, -1, axis=1)  # 有些年份第366项是0，所以去除

        # 计算dsl等并存储
        dso = np.zeros(40,dtype=int)
        dse = np.zeros(40,dtype=int)
        dsl = np.zeros(40,dtype=int)
        for year in range(40):
            p_ana = np.zeros(365)
            for i in range(365):
                p_ana[i] = np.sum(precip_year[year,0:i + 1]) - (i + 1) * p_e_mean

            max_index=signal.argrelmax(p_ana[0:120])
            if len(max_index[0])==0:
                k = np.argwhere(p_ana[0:120] == np.amax(p_ana[0:120]))
                dso[year] = k[0][0]
            else:
                dso[year]=max_index[0][0]
                for i in range(len(max_index[0])):
                    if p_ana[max_index[0][i]]>p_ana[dso[year]]:
                        dso[year]=max_index[0][i]

            min_index=signal.argrelmin(p_ana[dsl_mean:dsl_mean+120])
            min_index=np.array(min_index)
            min_index=min_index+dsl_mean
            if len(min_index[0])==0:
                e = np.argwhere(p_ana[dsl_mean:dsl_mean+120] == np.amin(p_ana[dsl_mean:dsl_mean+120]))+dsl_mean
                dse[year] = e[0][0]
            else:
                dse[year]=min_index[0][0]
                for i in range(len(min_index[0])):
                    if p_ana[min_index[0][i]]<p_ana[dse[year]]:
                        dse[year]=min_index[0][i]

            dsl[year] = dse[year] - dso[year]
            dso[year] = dso[year]
            dse[year] = dse[year]


        if dsl.min()<0:
            shutil.copy(e_path +"\\"+ file, data_path + "\\" + "e_error" + "\\" + file)
            continue

        dso_list[num, :] = dso[:]
        dse_list[num, :] = dse[:]
        dsl_list[num, :] = dsl[:]
        dso_mean_list[num,0]=dso_mean
        dso_mean_list[num,1]=p_e_mean
        dan_name[num] = file[:-4]
        num = num + 1

        # 写入行列号并保存至硬盘
        year = np.arange(1980, 2020)

        dso_list_df = pd.DataFrame(dso_list)
        dso_list_df.index = dan_name
        dso_list_df.columns = year
        dse_list_df = pd.DataFrame(dse_list)
        dse_list_df.index = dan_name
        dse_list_df.columns = year
        dsl_list_df = pd.DataFrame(dsl_list)
        dsl_list_df.index = dan_name
        dsl_list_df.columns = year
        dso_mean_list_df=pd.DataFrame(dso_mean_list)
        dso_mean_list_df.index=dan_name

        writer = pd.ExcelWriter(data_path+"\\"+"e_result\\e_dsodsedsl.xlsx")
        dso_list_df.to_excel(writer, sheet_name="dso")
        dse_list_df.to_excel(writer, sheet_name="dse")
        dsl_list_df.to_excel(writer, sheet_name="dsl")
        dso_mean_list_df.to_excel(writer, sheet_name="dso_mean")
        writer.save()
        writer.close()



