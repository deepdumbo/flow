from flow.models.u_net import UNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet()
model_path = 'C:/Users/Chris/flow/saved_models/sm.tar'
model.to(device)
