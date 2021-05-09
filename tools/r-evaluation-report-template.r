
library(ggplot2)

models = c(<MODELS>)
evalTypes = c(<EVALTYPES>)
evalSubtypes = c(<EVALSUBTYPES>)
values = c(<VALUES>)
datasets = c(<DATASETS>)

data.eval = data.frame(model=models, evalType=evalTypes, evalSubtype=evalSubtypes, value=values, dataset=datasets)

total <- split(data.eval, data.eval$evalType)

create_plot <- function(frame, ncol=3) {
    ggplot(frame, aes(value, model)) +
    facet_wrap(vars(evalType, evalSubtype, dataset), ncol=ncol, as.table = FALSE) + geom_point()
}