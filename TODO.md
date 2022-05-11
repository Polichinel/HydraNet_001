# TODO

- [ ] Impliement wasserstain distance as regression metric!

https://stats.stackexchange.com/questions/295617/what-is-the-advantages-of-wasserstein-metric-compared-to-kullback-leibler-diverg/295729#295729

https://arxiv.org/abs/1701.07875

- [ ] Impliment Multi-head model (classification and regression)
  - I guess you just have to normalize both loses before adding them togheter. But how to do that when the loss i just one number? Sigmoid the regression output?    
    > loss = loss1+loss2
    > loss.backward()
    > optimizer.step()
  - https://discuss.pytorch.org/t/how-to-combine-multiple-criterions-to-a-loss-function/348/14
  - Kendall et al. 2018
  - Note here that you actully do not have a normal regression task. Your measures as no negative and really the log of a poisson or power disttribution..
  - https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html#torch.nn.PoissonNLLLoss
  - https://pytorch.org/docs/stable/distributions.html?highlight=poisson#torch.distributions.poisson.Poisson
  https://pytorch.org/docs/stable/distributions.html#negativebinomial
  - Likly negative binomial is better then Poisson since it have no assumbtion regarding mean = variance.
  - https://pytorch.org/docs/stable/distributions.html#lognormal
  - But log-normal might be better since you do not really have count data after the log transformation..
  - But the question is, wheter you should do zero inflated, rather than multy task. But htne you can't really do the risk thing.. It is down to the question if it is really two different taks (perfctly corrolated) or two different task..

- [x] Random size of sampled subregions
- [ ] Use ViEWS replication Data - but you should also have a global model
- [ ] Now you also need to predict longer into the furture.
- [ ] Check if LTSM models are better now you use monthly data.
- [ ] Data uncertainty by using all three measure - or a distribution hereof.
- [ ] Model uncertinty by Prob. UNet.
- [ ] Add more channels: night light etc.

note every thing can be run from anaconda3/2021.05