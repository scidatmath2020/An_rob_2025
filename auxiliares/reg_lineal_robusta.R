library(robustbase)
library(MASS)
library(L1pack)

cisne = read.csv("cisne.csv")

lm.cisne = lm(log.light~log.Te,data=cisne)
summary(lm.cisne)
lm.cisne$coefficients

lts.cisne = ltsReg(log.light~log.Te,data=cisne)
summary(lts.cisne)
lts.cisne$coefficients

lms.cisne = lqs(log.light~log.Te,data=cisne,
                method ="lms")
lms.cisne$coefficients

lad.cisne = l1fit(cisne$log.Te,cisne$log.light)
lad.cisne$coefficients

lm.cisne$coefficients
lts.cisne$coefficients
lms.cisne$coefficients
lad.cisne$coefficients

df_coef <- data.frame(
  Model = c("OLS", "LTS", "LMS", "LAD"),
  Intercept = c(lm.cisne$coefficients[1], 
                lts.cisne$coefficients[1], 
                lms.cisne$coefficients[1], 
                lad.cisne$coefficients[1]),
  Slope = c(lm.cisne$coefficients[2], 
            lts.cisne$coefficients[2], 
            lms.cisne$coefficients[2], 
            lad.cisne$coefficients[2])
)

ggplot() + 
  geom_point(data=cisne,mapping=aes(x=log.Te,y=log.light)) +
  geom_abline(data=df_coef,mapping=aes(slope=Slope,intercept=Intercept,color=Model))

