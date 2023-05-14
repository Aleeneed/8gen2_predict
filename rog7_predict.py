import joblib
import random
import pandas as pd
import numpy as np
def do_task(time,up0=100,up3=100,up7=100,c0_frq=2016,c3_frq=2803,c7_frq=3187):
    cpu0=[]
    cpu3=[]
    cpu7=[]
    cpu0_frq=[]
    cpu3_frq=[]
    cpu7_frq=[]
    Power=[]
    temps=[]
    frs=[]
    for i in range(time):
        cpu0_usage=random.randint(10,up0)
        cpu3_usage=random.randint(10,up3)
        cpu7_usage=random.randint(1,up7)
        cpu0_Frq=random.randint(1017,c0_frq)
        cpu3_Frq=random.randint(998,c3_frq)
        cpu7_Frq=random.randint(864,c7_frq)
        req_power=joblib.load("./ROG7_Predict/rog7_60hz_power.pkl")
        req_Temp=joblib.load("./ROG7_Predict/rog7_60hz_Temp.pkl")
        req_fps=joblib.load("./ROG7_Predict/rog7_60hz_Fps.pkl")    
        cpu=[[cpu0_usage,cpu0_usage,cpu0_usage,cpu3_usage,cpu3_usage,cpu3_usage,cpu3_usage,cpu7_usage,cpu0_Frq,cpu3_Frq,cpu7_Frq]]
        cpu_0=[cpu0_usage]
        cpu_3=[cpu3_usage]
        cpu_7=[cpu7_usage]
        cpu_0_frq=[cpu0_Frq]
        cpu_3_frq=[cpu3_Frq]
        cpu_7_frq=[cpu7_Frq]
        cpu0.append(cpu_0)
        cpu3.append(cpu_3)
        cpu7.append(cpu_7)
        cpu0_frq.append(cpu_0_frq)
        cpu3_frq.append(cpu_3_frq)
        cpu7_frq.append(cpu_7_frq)
        power=req_power.predict(cpu)
        Temp=req_Temp.predict(cpu)
        fps=req_fps.predict(cpu)
        Power.append(power)
        temps.append(Temp)
        frs.append(fps)
    ca=[]
    ca3=[]
    ca7=[]
    frq0=[]
    frq3=[]
    frq7=[]
    p=[]
    t=[]
    f=[]
    for i in cpu0:
        ca.append(i)
    for i in cpu3:
        ca3.append(i)    
    for i in cpu7:
        ca7.append(i)
    for i in cpu0_frq:
        frq0.append(i)   
    for i in cpu3_frq:
        frq3.append(i)
    for i in cpu7_frq:
        frq7.append(i)   
    for i in Power:
        p.append(i)
    for i in temps:
        t.append(i)    
    for i in frs:
        f.append(i)
    ca=list(np.ravel(ca))
    ca3=list(np.ravel(ca3))
    ca7=list(np.ravel(ca7))
    frq0=list(np.ravel(frq0))
    frq3=list(np.ravel(frq3))
    frq7=list(np.ravel(frq7))  
    p=list(np.ravel(p))
    t=list(np.ravel(t))
    f=list(np.ravel(f))
    data={"CPU 0 Usage":ca,"CPU 3 Usage":ca3,"CPU 7 Usage":ca7,"CPU0_Frq":frq0,"CPU3_Frq":frq3,"CPU7_Frq":frq7,"FPS":f,"TEMP":t,"Power":p} 
    data_df=pd.DataFrame(data)
    new=data_df["FPS"].mean()
    new1=[]
    new1.append(new)
    std1=data_df["FPS"].std()
    std11=[]
    std11.append(std1)
    new_1={"平均FPS":new1,"標準差":std11}
    te=data_df["TEMP"].mean()
    te1=[]
    te1.append(te)
    te_1={"平均溫度":te1}
    po=data_df["Power"].mean()
    po1=[]
    po1.append(po)
    po_1={"平均功耗":po1}
    new_df=pd.DataFrame(new_1)
    new1_df=pd.DataFrame(te_1)
    new2_df=pd.DataFrame(po_1)
    mix_df=pd.concat([data_df,new_df,new1_df,new2_df],axis=1)
    mix_df.to_csv("./ROG7_Predict.csv",index=False,encoding='utf_8_sig')
if __name__=="__main__":
    do_task(10000,80,100,1,1804,2745,864) 
