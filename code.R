rm(list = ls())
mydata <- read.delim("clipboard")#导入剪贴板中的数据
library(compareGroups)
library(glue)
library(CBCgrps)
library(nortest)
library(flextable)
library(xtable)
library(plyr)
str(mydata)
#更改数据类型
mydata[,c("Marital","Race","Sequence_number","Gender","Laterality",'Grade','Primary_Site','Surgery','Radiation',
          "Chemotherapy","System_management",'T','N','M','Age')] <- 
  lapply(mydata[,c("Marital","Race","Sequence_number","Gender","Laterality",'Grade','Primary_Site','Surgery','Radiation',
                   "Chemotherapy","System_management",'T','N','M','Age')], factor)
str(mydata)
summary(mydata)
# descrTable(hypertension ~ ., data = data)
# cGroupsGUI(data.frame(mydata))
table1 <- twogrps(mydata,
                  gvar = "M",
                  minfactorlevels = 10,
                  maxfactorlevels = 10)
print(table1,quote = T,noSpace = T)
#cGroupsGUI(data.frame(mydata))
table_one <- descrTable(M~ ., data = mydata)####.符号代表包括其他的变量
getwd()
setwd('E:\\Spyder_2022.3.29\\output\\machinel\\lwl_output\\NM_WSN')
getwd()
export2csv(table_one,file = "table1.csv")
table_all <- descrTable(~.,data = mydata)
export2csv(table_all,file = "table1ll.csv")

# Logistic回归 --------------------------------------------------------------
aa <- mydata
#分类变量一定要变为分类形式
# for (i in names(aa)[c(1:19)]) {aa[,i] <- as.factor(aa[,i])}
str(aa)

Uni_glm_model<- 
  function(x){
    #拟合结局和变量
    FML<-as.formula(paste0("M==1~",x))
    #glm()逻辑回归
    glm1<-glm(FML,data=aa,family = binomial)
    #提取所有回归结果放入glm2中
    glm2<-summary(glm1)
    #1-计算OR值保留两位小数
    OR<-round(exp(coef(glm1)),3)
    #2-提取SE
    SE<-glm2$coefficients[,2]
    #3-计算CI保留两位小数并合并
    CI5<-round(exp(coef(glm1)-1.96*SE),3)
    CI95<-round(exp(coef(glm1)+1.96*SE),3)
    CI<-paste0(CI5,'-',CI95)
    OR95 <- paste0(OR," ( ",CI," ) ")
    #4-提取P值
    P<-round(glm2$coefficients[,4],3)
    #5-将变量名、OR、CI、P合并为一个表，删去第一行
    Uni_glm_model <- data.frame('Characteristics'=x,
                                'OR' = OR,
                                '95%CI' = CI,
                                "OR95" = OR95,
                                'Pvalue' = P)[-1,]
    #返回循环函数继续上述操作                     
    return(Uni_glm_model)
  }  
#x取值的确定
names(aa)
variable.names <- colnames(aa)[c(1:14)];variable.names
Uni_glm <- lapply(variable.names, Uni_glm_model)
res_uni <- ldply(Uni_glm, data.frame);res_uni
#p四舍五入为0的改为<0.001
res_uni$Pvalue[res_uni$Pvalue ==0] <- '<0.001'
View(res_uni)#查看单因素回归结果

#2. P<0.05进入多因素循环
#提取单因素p<0.05的变量
res_uni$Characteristics[res_uni$Pvalue<0.05]
#将其纳入多因素模型
mul_glm_model <- as.formula(paste0("M==1~",
                                   paste0(res_uni$Characteristics[res_uni$Pvalue<0.05],
                                          collapse = "+")))
mul_glm <- glm(mul_glm_model,
               family = binomial(link = 'logit'),
               data = aa)
glm3 <- summary(mul_glm)
summary(mul_glm)
#提取多因素回归的结果
mul_OR <- round(exp(glm3$coefficients[,1]),3)
mul_SE <- glm3$coefficients[,2]
mul_CI1 <- round(exp(glm3$coefficients[,1]-1.96*mul_SE),3)
mul_CI2 <- round(exp(glm3$coefficients[,1]+1.96*mul_SE),3)
mul_CI <- paste0(mul_CI1,'-',mul_CI2)
mul_OR95 <- paste0(mul_OR," ( ",mul_CI," ) ")
mul_P <- round(glm3$coefficients[,4],3)
#将提取的结果整合成表
res_mul <- data.frame("OR2" =mul_OR,
                      "95%CI2" =mul_CI,
                      "mul_OR95" = mul_OR95,
                      "Pvalue2" = mul_P)[-1,]
#将P值四舍五入改为<0.001
res_mul$Pvalue2[res_mul$Pvalue2== 0] <- '<0.001'
#单因素和多因素的合并
# #删除单因素表的第一列
# res_uni <- res_uni[,-1]
# #新的第一列命名为
# colnames(res_uni)[1] <- 'Characteristics'
# #多因素的表的行名放入单元格中，也命名为Characteristics
res_mul <- cbind(rownames(res_mul),
                 res_mul, row.names=NULL)
names(res_mul)[1] <- "Characteristics"
#二表合并
table2 <- merge.data.frame(res_uni, res_mul,
                           by = 'Characteristics',
                           all =T,
                           sort = F)
#查看最终的表格
View(table2)
print(table2,quote = T)
getwd()
setwd("E:\\Spyder_2022.3.29\\output\\machinel\\lwl_output\\OSA")
getwd()
write.csv(table2, file = "table2.csv")
getwd()

#显示纳入机器学习模型的变量名字
res_uni$Characteristics[res_uni$Pvalue<0.05]


#绘制雷达图
library(ggradar)
library(palmerpenguins)
library(tidyverse)
library(scales)
library(showtext)
font_add_google("Lobster Two", "lobstertwo")
font_add_google("Roboto", "roboto")
# Showtext will be automatically invoked when needed
showtext_auto()
rm(list = ls())
#load data
data <- read.delim("clipboard")#导入剪贴板中的数据
plt <- data %>%
  ggradar(
    font.radar = "roboto",
    grid.label.size = 10,  # Affects the grid annotations (0%, 50%, etc.)
    axis.label.size = 7, # Afftects the names of the variables
    group.point.size = 3   # Simply the size of the point 
  )
plt
plt <- plt + 
  theme(
    legend.position = c(1, 0),  
    legend.justification = c(1, 0),
    legend.text = element_text(size = 20, family = "roboto"),
    legend.key = element_rect(fill = NA, color = NA),
    legend.background = element_blank()
  )
plt
# * The panel is the drawing region, contained within the plot region.
#   panel.background refers to the plotting area
#   plot.background refers to the entire plot
plt <- plt + 
  labs(title = "Radar plot of Six machine learning Methods") + 
  theme(
    plot.background = element_rect(fill = "#fbf9f4", color = "#fbf9f4"),
    panel.background = element_rect(fill = "#fbf9f4", color = "#fbf9f4"),
    plot.title.position = "plot", # slightly different from default
    plot.title = element_text(
      family = "Roboto", 
      size = 30,
      face = "bold", 
      color = "#2a475e"
    )
  )
plt

####单因素多因素COx
#单因素多因素cox回归表格
rm(list = ls())
library(survival)
library(plyr)
aa <- read.delim("clipboard")
str(aa)
aa[,c("Marital","Race","Sequence_number","Gender","Laterality",'Grade','Primary_Site','Surgery','Radiation',
      "Chemotherapy","System_management",'T','N','Age')] <- 
  lapply(aa[,c("Marital","Race","Sequence_number","Gender","Laterality",'Grade','Primary_Site','Surgery','Radiation',
               "Chemotherapy","System_management",'T','N','Age')], factor)
str(aa)
#查看结局
aa$fustat <- factor(aa$fustat)
summary(aa$fustat)
#构建模型的y
y <- Surv(time = aa$futime, event = aa$fustat ==1)#1为死亡
#批量构建单因素cox回归
Uni_cox_model <- function(x){
  FML <- as.formula(paste0("y~",x))
  cox <- coxph(FML, data = aa)
  cox1 <- summary(cox)
  HR <- round(cox1$coefficients[,2],3)
  PValue <- round(cox1$coefficients[,5],3)
  CI5 <- round(cox1$conf.int[,3],3)
  CI95 <- round(cox1$conf.int[,4],3)
  CI <- paste0(CI5,'-',CI95)
  HR95 <- paste0(HR," ( ",CI," ) ")
  Uni_cox_model <- data.frame('Characteristics' = x,
                              "HR_Uni" = HR,
                              "CI5" = CI5,
                              "CI95" = CI95,
                              'p' = PValue)
  return(Uni_cox_model)
}
#将想要进行的单因素回归变量输入模型
names(aa)
#输入变量序号
variable.names <- colnames(aa)[c(1:15)]

#输出结果
Uni_cox <- lapply(variable.names,Uni_cox_model)
Uni_cox <- ldply(Uni_cox, data.frame)
#优化表格
Uni_cox$CI_Uni <- paste(Uni_cox$CI5, '-', Uni_cox$CI95)
Uni_cox$P_value1 <- paste(Uni_cox$p)
Uni_cox <- Uni_cox[,-3:-5]
#将P值四舍五入改为<0.001
Uni_cox$P_value1[Uni_cox$P_value1== 0] <- '<0.001'
View(Uni_cox)

#单因素回归纳入多因素回归
Uni_cox$Characteristics[Uni_cox$P_value1<0.05]
#多因素模型建立
mul_cox_model <- as.formula(paste0('y~',
                                   paste0(Uni_cox$Characteristics[Uni_cox$P_value1<0.05],
                                          collapse = "+")))
mul_cox <- coxph(mul_cox_model, data =aa)
cox4 <- summary(mul_cox)

#提取多因素回归的信息
mul_HR <- round(cox4$coefficients[,2],3)
mul_PValue <- round(cox4$coefficients[,5],3)
mul_CI1 <- round(cox4$conf.int[,3],3)
mul_CI2 <- round(cox4$conf.int[,4],3)
#多因素结果优化为表格
mul_CI <- paste(mul_CI1,'-',mul_CI2)
mul_cox1 <- data.frame("HR_Mul" = mul_HR,
                       'CI_Mul' = mul_CI,
                       'P_value2' = mul_PValue)
#将P值四舍五入改为<0.001
mul_cox1$P_value2[mul_cox1$P_value2== 0] <- '<0.001'
#单因素和多因素的合并
# #删除单因素表的第一列
# res_uni <- res_uni[,-1]
# #新的第一列命名为
# colnames(res_uni)[1] <- 'Characteristics'
# #多因素的表的行名放入单元格中，也命名为Characteristics
mul_cox1 <- cbind(rownames(mul_cox1),
                  mul_cox1, row.names=NULL)
names(mul_cox1)[1] <- "Characteristics"
#二表合并
table2 <- merge.data.frame(Uni_cox, mul_cox1,
                           by = 'Characteristics',
                           all =T,
                           sort = T)
#查看最终的表格
View(table2)
print(table2,quote = T)
#保存为csv
write.csv(table2, file = "table2.csv")
getwd()



#森林图的绘制

rm(list = ls())     
outFile="forest_mul.pdf"         
# setwd("E:\\Spyder_2022.3.29\\output\\machinel\\lwl_output\\NOS_WSN")    


rt=read.delim("clipboard")
rownames(rt) = rt[,1]
rt = rt[,-1]
gene=rownames(rt)
hr=sprintf("%.3f",rt$"HR")
hrLow=sprintf("%.3f",rt$"HR.95L")
hrHigh=sprintf("%.3f",rt$"HR.95H")
Hazard.ratio=paste0(hr,"(",hrLow,"-",hrHigh,")")
pVal=rt$pvalue

#出图格式
pdf(file=outFile, width = 8.5, height =5)
n=nrow(rt)
nRow=n+1
ylim=c(1,nRow)
layout(matrix(c(1,2),nc=2),width=c(3,2))

#森林图左边的基因信息
xlim = c(0,3)
par(mar=c(4,2,1.5,1.5))
plot(1,xlim=xlim,ylim=ylim,type="n",axes=F,xlab="",ylab="")
text.cex=0.8
text(0,n:1,gene,adj=0,cex=text.cex)
text(1.5-0.5*0.2,n:1,pVal,adj=1,cex=text.cex);text(1.5-0.5*0.2,n+1,'P value',cex=text.cex,font=2,adj=1)
text(3,n:1,Hazard.ratio,adj=1,cex=text.cex);text(3,n+1,'Hazard ratio',cex=text.cex,font=2,adj=1,)

#绘制森林图
par(mar=c(4,1,1.5,1),mgp=c(2,0.5,0))
xlim = c(0,max(as.numeric(hrLow),as.numeric(hrHigh)))
plot(1,xlim=xlim,ylim=ylim,type="n",axes=F,ylab="",xaxs="i",xlab="Hazard ratio")
arrows(as.numeric(hrLow),n:1,as.numeric(hrHigh),n:1,angle=90,code=3,length=0.03,col="darkblue",lwd=2.5)
abline(v=1,col="black",lty=2,lwd=2)
boxcolor = ifelse(as.numeric(hr) > 1, 'red', 'blue')
points(as.numeric(hr), n:1, pch = 15, col = boxcolor, cex=1.3)
axis(1)
dev.off()


#列线图的绘制

# install.packages("rms")

rm(list = ls())
library(rms)                
rt=read.delim("clipboard")
# rt$Marital<- factor(rt$Marital, levels= c(0,1), labels= c("否","是"))
#数据打包
dd <- datadist(rt)
options(datadist="dd")
#生成函数
f <- cph(Surv(futime, fustat) ~ MLP+Age+Chemotherapy+Laterality+Marital+N+Radiation
         +Surgery+T+Sequence_number+System_management, 
         x=T, y=T, surv=T, data=rt, time.inc=1)
surv <- Survival(f)
#建立nomogram
nom <- nomogram(f, fun=list(function(x) surv(1, x), function(x) surv(3, x), function(x) surv(5, x)), 
                lp=F, funlabel=c("1-year survival", "3-year survival", "5-year survival"), 
                maxscale=100, 
                fun.at=c(0.99, 0.9, 0.8,0.7,0.3, 0.5,0.1,0.01))  

#nomogram可视化
pdf(file="Nomogram.pdf",height=10,width=12)
plot(nom)
dev.off()


#calibration curve
time=1   #预测三年calibration
f <- cph(Surv(futime, fustat) ~ MLP+Age+Chemotherapy+Laterality+Marital+N+Radiation
         +Surgery+T+Sequence_number+System_management,
         x=T, y=T, surv=T, data=rt, time.inc=1)
cal <- calibrate(f, cmethod='KM', method="boot", u=time, m=1000, B=50) #m样品数目1/3
pdf(file="calibration_1year.pdf",height=6,width=7)
plot(cal,
     # xlim = c(0,1),ylim = c(0,1),
     xlab="Nomogram-Predicted Probability of 1-Year OS",
     ylab="Actual 1-Year OS(proportion)",
     col="red",
     sub=F)
dev.off()

#calibration curve
time=1.1   #预测三年calibration
# f <- cph(Surv(times, status) ~BAG+gender+N+Primary.Site+Race+
#            Radiation+Sequence.number+surgery, x=T, y=T, surv=T, data=rt, time.inc=time)
cal <- calibrate(f, cmethod="KM", method="boot", u=time, m=1000, B=50) #m样品数目1/3
pdf(file="calibration_3years.pdf",height=6,width=7)
plot(cal,
     # xlim = c(0,1),ylim = c(0,1),
     xlab="Nomogram-Predicted Probability of 3-Year OS",
     ylab="Actual 3-Year OS(proportion)",
     col="red",
     sub=F)
dev.off()

#calibration curve
time=1.15   #预测三年calibration
# f <- cph(Surv(times, status) ~BAG+gender+N+Primary.Site+Race+
#            Radiation+Sequence.number+surgery, x=T, y=T, surv=T, data=rt, time.inc=time)
cal <- calibrate(f, cmethod="KM", method="boot", u=time, m=1000, B=50) #m样品数目1/3
pdf(file="calibration_5years.pdf",height=6,width=7)
plot(cal,
     # xlim = c(0,1),ylim = c(0,1),
     xlab="Nomogram-Predicted Probability of 5-Year OS",
     ylab="Actual 5-Year OS(proportion)",
     col="red",
     sub=F)
dev.off()



#TimeROC曲线的绘制
rm(list = ls())
library(survival)
library(survminer)
library(timeROC)
outFile="ROC.pdf"         
var="score"               
rt=read.delim("clipboard")

#绘制
ROC_rt=timeROC(T=rt$futime, delta=rt$fustat,
               marker=rt[,var], cause=1,
               weighting='aalen',
               times=c(1,2,3), ROC=TRUE)
pdf(file=outFile,width=5,height=5)
plot(ROC_rt,time=1,col='green',title=FALSE,lwd=2)
plot(ROC_rt,time=2,col='blue',add=TRUE,title=FALSE,lwd=2)
plot(ROC_rt,time=3,col='red',add=TRUE,title=FALSE,lwd=2)
legend('bottomright',
       c(paste0('AUC at 1 years: ',sprintf("%.03f",ROC_rt$AUC[1])),
         paste0('AUC at 2 years: ',sprintf("%.03f",ROC_rt$AUC[2])),
         paste0('AUC at 3 years: ',sprintf("%.03f",ROC_rt$AUC[3]))),
       col=c("green",'blue','red'),lwd=2,bty = 'n')
dev.off()


#绘制KM曲线
rm(list = ls())
library(survival)
library(survminer)
rt=read.delim("clipboard")
rt[,c("Marital","Race","Sequence_number","Gender","Laterality",'Grade','Primary_Site','Surgery','Radiation',
      "Chemotherapy","System_management",'T','N','Age')] <- 
  lapply(rt[,c("Marital","Race","Sequence_number","Gender","Laterality",'Grade','Primary_Site','Surgery','Radiation',
               "Chemotherapy","System_management",'T','N','Age')], factor)
# inputFile="input.txt"         
outFile="survival_Primary_Site.pdf"        
var="Primary_Site"                   #用于生存分析的变量
#读取
# rt$Marital <- factor(rt$Marital, levels = c(0,1,2), labels = c("Married","unmarried",'unknown '))
# rt$Race <- factor(rt$Race, levels = c(0,1,2,3), labels = c("white","black",'Chinese',"other"))
# rt$gender <- factor(rt$gender, levels = c(0,1), labels = c("Male","Female",))
# rt$N <- factor(rt$gender, levels = c(0,1), labels = c("Male","Female",))
rt=rt[,c("futime","fustat",var)]

colnames(rt)[3]="Type"
groupNum=length(levels(factor(rt[,"Type"])))

#比较组间生存差异的P值
diff=survdiff(Surv(futime, fustat) ~Type,data = rt)
pValue=1-pchisq(diff$chisq,df=(groupNum-1))  #df自由度
if(pValue<0.001){
  pValue="p<0.001"
}else{
  pValue=paste0("p=",sprintf(".03f",pValue))
}
fit <- survfit(Surv(futime, fustat) ~ Type, data = rt)
pValue
#绘制
surPlot=ggsurvplot(fit, 
                   data=rt,
                   conf.int=F,  #置信区间
                   pval=pValue,
                   pval.size=5,
                   legend.labs=levels(factor(rt[,"Type"])),
                   legend.title="Group",
                   xlab="Time(Years)",
                   break.time.by = 2,
                   risk.table.title=var,
                   risk.table=T,
                   risk.table.col = "strata", # 根据分层更改风险表颜色
                   risk.table.height=.25,
                   ggtheme = theme_bw()) # 添加中位生存时间线
pdf(file=outFile,onefile = FALSE,width = 8,height =14)
print(surPlot)
dev.off()


#模型的构建
library(survival)                                             
rt <- read.delim("clipboard")
multiCox=coxph(Surv(futime, fustat) ~ ., data = rt)
multiCox=step(multiCox,direction = "both")
multiCoxSum=summary(multiCox)

#????ģ?Ͳ???
outTab=data.frame()
outTab=cbind(
  coef=multiCoxSum$coefficients[,"coef"],
  HR=multiCoxSum$conf.int[,"exp(coef)"],
  HR.95L=multiCoxSum$conf.int[,"lower .95"],
  HR.95H=multiCoxSum$conf.int[,"upper .95"],
  pvalue=multiCoxSum$coefficients[,"Pr(>|z|)"])
outTab=cbind(id=row.names(outTab),outTab)
outTab=gsub("`","",outTab)
write.table(outTab,file="multiCox.xls",sep="\t",row.names=F,quote=F)

#???????˷???ֵ
riskScore=predict(multiCox,type="risk",newdata=rt)
coxGene=rownames(multiCoxSum$coefficients)
coxGene=gsub("`","",coxGene)
outCol=c("futime","fustat",coxGene)
risk=as.vector(ifelse(riskScore>median(riskScore),"high","low"))
write.table(cbind(id=rownames(cbind(rt[,outCol],riskScore,risk)),cbind(rt[,outCol],riskScore,risk)),
            file="risk.txt",
            sep="\t",
            quote=F,
            row.names=F)


#135年的ROC曲线
# install.packages("scales")
rm(list= ls())
library(survivalROC)
library("scales")
require(ggsci)
library("scales")
pal_nejm("default")(8)
show_col(pal_nejm("default")(8))

rt=read.table("risk.txt",header=T,sep="\t",check.names=F,row.names=1)    #??ȡlasso?ع??????ļ?
pdf(file="ROC.pdf",width=6,height=6)
par(oma=c(0.5,1,0,1),font.lab=1.5,font.axis=1.5)
roc=survivalROC(Stime=rt$futime, status=rt$fustat, marker = rt$riskScore, 
                predict.time =5, method="KM")
plot(roc$FP, roc$TP, type="l", xlim=c(0,1), ylim=c(0,1),col='#BC3C29FF', 
     xlab="False positive rate", ylab="True positive rate",
     main=paste("ROC curve"),
     lwd = 2, cex.main=1.3, cex.lab=1.2, cex.axis=1.2, font=1.2)


roc1=survivalROC(Stime=rt$futime, status=rt$fustat, marker = rt$riskScore, 
                 predict.time =3, method="KM")

lines(roc1$FP, roc1$TP, type="l",col="#0072B5FF",xlim=c(0,1), ylim=c(0,1))

roc2=survivalROC(Stime=rt$futime, status=rt$fustat, marker = rt$riskScore, 
                 predict.time =1, method="KM")

lines(roc2$FP, roc2$TP, type="l",col="#EE4C97FF",xlim=c(0,1), ylim=c(0,1))

legend(0.6,0.2,c(paste("AUC of 5 year = ",round(roc$AUC,3)),
                 paste("AUC of 3 year = ",round(roc1$AUC,3)),
                 paste("AUC of 1 year = ",round(roc2$AUC,3))),
       x.intersp=1, y.intersp=0.8,
       lty= 1 ,lwd= 2,col=c("#BC3C29FF","#0072B5FF","#EE4C97FF"),
       bty = "n",# bty框的类型
       seg.len=1,cex=0.8)# 

abline(0,1,col="gray",lty=2)
dev.off()

#风险曲线的绘制
library(pheatmap)
rt=read.table("risk.txt",sep="\t",header=T,row.names=1,check.names=F)       #??ȡ?????ļ?
rt=rt[order(rt$riskScore),]                                     #????riskScore????????

#???Ʒ???????
riskClass=rt[,"risk"]
lowLength=length(riskClass[riskClass=="low"])
highLength=length(riskClass[riskClass=="high"])
line=rt[,"riskScore"]
line[line>10]=10
pdf(file="riskScore.pdf",width = 10,height = 3.5)
plot(line,
     type="p",
     pch=20,
     xlab="Patients (increasing risk socre)",
     ylab="Risk score",
     col=c(rep("green",lowLength),
           rep("red",highLength)))
abline(h=median(rt$riskScore),v=lowLength,lty=2)
legend("topleft", c("High risk", "low Risk"),bty="n",pch=19,col=c("red","green"),cex=1.2)
dev.off()

#????????״̬ͼ
color=as.vector(rt$fustat)
color[color==1]="red"
color[color==0]="green"
pdf(file="survStat.pdf",width = 10,height = 3.5)
plot(rt$futime,
     pch=19,
     xlab="Patients (increasing risk socre)",
     ylab="Survival time (years)",
     col=color)
legend("topleft", c("Dead", "Alive"),bty="n",pch=19,col=c("red","green"),cex=1.2)
abline(v=lowLength,lty=2)
dev.off()

#???Ʒ?????ͼ
rt1=log2(rt[c(3:(ncol(rt)-2))]+1)
rt1=t(rt1)
annotation=data.frame(type=rt[,ncol(rt)])
rownames(annotation)=rownames(rt)
pdf(file="heatmap.pdf",width = 12,height =4)
pheatmap(rt1, 
         annotation=annotation, 
         cluster_cols = FALSE,
         fontsize_row=11,
         show_colnames = F,
         fontsize_col=3,
         color = colorRampPalette(c("green", "black", "red"))(50) )
dev.off()


#DCA曲线的绘制
rm(list = ls())
library(survival)
library(ggDCA)
riskFile="risk.txt"         #?????????ļ?
#??ȡ?????????ļ?
risk=read.table(riskFile, header=T, sep="\t", check.names=F, row.names=1)

rt = risk

predictTime=1   
Risk<-coxph(Surv(futime,fustat)~risk,rt)
Marital<-coxph(Surv(futime,fustat)~Marital,rt)
Age<-coxph(Surv(futime,fustat)~Age,rt)
Laterality<-coxph(Surv(futime,fustat)~Laterality,rt)
MLP<-coxph(Surv(futime,fustat)~MLP,rt)
N<-coxph(Surv(futime,fustat)~N,rt)
pdf(file="DCA.pdf", width=6.5, height=5.2)
d_train=dca(Risk,Marital,Age,Laterality,MLP,N,times=predictTime)
ggplot(d_train, linetype=1)
dev.off()








