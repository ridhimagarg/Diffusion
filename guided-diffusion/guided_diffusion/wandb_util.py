import wandb



def wandb_setup(group_name):
    # return wandb.init(project="Diffusion-Ridhima", entity="robofied")
    return wandb.init(project="Diffusion", entity="team_uni_stuttgart", group=group_name)
    # WANDB_API_KEY = 'fa8770add2e41bde69cf6061e33e4c51e9cdd676'
    # WANDB_API_KEY = '086e3f5d98b58ab5e34f2814915b14c0ab230bfc'
