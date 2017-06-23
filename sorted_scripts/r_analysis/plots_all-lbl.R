require(ggplot2)
require(dplyr)
require(viridis)
require(Rmisc)

dat <- read.csv("power_data_no-step_both_grps_all-freqs_mean.csv")
data_clean <- subset(dat, trial_status == "True")
data_sub <- subset(data_clean, band != "theta" & band != "gamma2")
data_sub$band <- factor(levels(droplevels(data_sub$band)))

data_sub$label <- droplevels(data_sub$label)

data_mean <- tbl_df(data_sub) %>% group_by(subject, group) %>% summarise(r = mean(r), binding=mean(binding))

print(ggplot(data_mean, aes(y=r, x=binding, color=group)) +
          scale_colour_hue(l=50) + # Use a slightly darker palette than normal
          geom_point(shape=16, size=3) +
          geom_smooth(method=lm,   # Add linear regression lines
                      se=TRUE,    # Don't add shaded confidence region
                      fullrange=TRUE) +
              # facet_wrap(~label) +
          scale_colour_manual(values= c('red', 'blue')) +
          theme(legend.position="bottom") +
          ggtitle("power") +
          ggsave("~/projects/agency_connectivity/data/plots/power_-lbl_ib-r.png",
                    dpi = 400, height = 10, width = 15))


data_mean <- read.csv("power_data_no-step_both_grps_all-freqs_no-lbl_mean.csv")
data_mean <- subset(data_clean, band != "theta" & band != "gamma2")
data_mean$band <- factor(levels(droplevels(data_sub$band)))
data_summary <- summarySE(data_mean, measurevar="binding",
                          groupvars=c("group", "band"))

ggplot(data_summary, aes(x=group, y=binding, colour=band)) + 
    geom_bar(position=position_dodge(), stat="identity", colour="Black") +
    geom_errorbar(aes(ymin=binding-ci, ymax=binding+ci),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
    scale_fill_manual(values=c("#CCCCCC","#FFFFFF", "#808080")) +    theme(legend.position="bottom") + 
    ggtitle("R for groups across bands")


dat <- read.csv("itc_data_no-step_both_grps_all-freqs_mean.csv")
data_clean <- subset(dat, trial_status == "True")
data_sub <- subset(data_clean, band != "theta" & band != "gamma2")
data_sub$band <- factor(levels(droplevels(data_sub$band)))

data_sub$label <- droplevels(data_sub$label)

data_mean <- tbl_df(data_sub) %>% group_by(subject, group) %>% summarise(ISPC = mean(ISPC
), binding=mean(binding))

print(ggplot(data_mean, aes(x=ISPC, y=binding, color=group)) +
          scale_colour_hue(l=50) + # Use a slightly darker palette than normal
          geom_point(shape=16, size=3) +
          geom_smooth(method=lm,   # Add linear regression lines
                      se=TRUE,    # Don't add shaded confidence region
                      fullrange=TRUE) +
          # facet_wrap(~label) +
          scale_colour_manual(values= c('red', 'blue')) +
          theme(legend.position="bottom") +
          ggtitle("ISPC") +
          labs(title = "ISPC",
               y = "Binding effect", x = "ISPC") + 
          ggsave("~/projects/agency_connectivity/data/plots/phase_band_all-lbl_ib-r.png",
                 dpi = 400, height = 10, width = 15))


dat <- read.csv("power_data_no-step_both_grps_all-freqs_mean.csv")
data_clean <- subset(dat, trial_status == "True")
data_sub <- subset(data_clean, band != "theta" & band != "gamma2")
data_sub$band <- factor(levels(droplevels(data_sub$band)))

data_mean <- tbl_df(data_sub) %>% group_by(subject, group, label) %>% summarise(r = mean(r), binding=mean(binding))

ggplot(data_mean, aes(y=group, x=label)) +
    geom_tile(aes(fill = r)) +
    scale_fill_viridis() +
    theme(text = element_text(size=20),
          axis.text.x = element_text(angle=90, hjust=1)) + 
    ggsave("~/projects/agency_connectivity/data/plots/power_heatmap_r_grp-lbl.png",
           dpi = 400, height = 10, width = 15)


dat <- read.csv("power_mean_group_lbl.csv")
dat_sel <- subset(dat, label=="ba_1-lh_39-rh" | label=="ba_1-rh_39-rh" |
                      label=="ba_4-rh_46-rh")

print(ggplot(dat_sel, aes(y=binding, x=r)) +
          geom_point(aes(color=label), size=3) +
          scale_colour_manual(values=c("red","blue", "black")) +
          geom_smooth(method=lm,   # Add linear regression lines
                      se=TRUE,    # Don't add shaded confidence region
                      fullrange=TRUE, aes(group=label, color=label)) +
          # facet_wrap(~label) +
          # scale_colour_manual(values= viridis) +
          theme(legend.position="bottom") + 
          labs(title = "Selected labels (t > 2)",
               y = "Binding effect", x = "Spearman's r")) 
          # ylim(-0.0, 0.25))

data_summary <- summarySE(dat_sel, measurevar="r",
                          groupvars=c("group", "label"))

ggplot(data_summary, aes(x=label, y=r, fill=group)) + 
      geom_bar(position=position_dodge(), stat="identity", colour="Black") +
      geom_errorbar(aes(ymin=r-ci, ymax=r+ci),
                    width=.2,                    # Width of the error bars
                    position=position_dodge(.9)) +
      scale_fill_manual(values=c("#CCCCCC","#808080")) +
      theme(legend.position="bottom")

dat <- read.csv("power_mean_group_band.csv")
data_summary <- summarySE(dat, measurevar="binding",
                              groupvars=c("group", "band"))
    
ggplot(data_summary, aes(x=band, y=binding, fill=group)) + 
    geom_bar(position=position_dodge(), stat="identity", colour="Black") +
    geom_errorbar(aes(ymin=binding-ci, ymax=binding+ci),
                  width=.2,                    # Width of the error bars
                  position=position_dodge(.9)) +
    scale_fill_manual(values=c("#CCCCCC","#808080")) +
    theme(legend.position="bottom")

dat <- read.csv("itc_mean_group_lbl.csv")
dat_sel <- subset(dat, label=="ba_1-rh_39-rh")

print(ggplot(dat_sel, aes(y=binding, x=ISPC)) +
          geom_point(aes(color=label), size=3) +
          scale_colour_manual(values=c("black")) +
          geom_smooth(method=lm,   # Add linear regression lines
                      se=TRUE,    # Don't add shaded confidence region
                      fullrange=TRUE, aes(group=label, color=label)) +
          # scale_colour_manual(values= viridis) +
          theme(legend.position="bottom") + 
          labs(title = "Selected labels (t > 2)",
               y = "ISPC", x = "Binding effect")
          # ylim(0.27, 0.35))

