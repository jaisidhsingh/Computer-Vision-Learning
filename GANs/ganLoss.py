import torch


class StandardGANLoss():
    def __init__(self, discrimator, generator, z_dim,batch_size, real, criterion):
        self.disc = discrimator
        self.gen = generator
        self.z_dim = z_dim
        self.real = real
        self.criterion = criterion


    def DiscLoss(self):
        """Tip for Usage : 

           discLoss, fake = StandardGANLoss(...).DiscLoss()
           discLoss.backward(retain_graph=True)
           genLoss = StandardGANLoss(...).GenLoss(fake) 

           Setting retain_graph = True saves computation
           while getting discrimator output during generator
           training.
        """
    
        noise = torch.randn((batch_size, self.z_dim))
        fake = self.gen(noise)
        discReal = disc(real).view(-1)
        discFake = disc(fake).view(-1)

        # corresponds to maxinmising log(D(real)) [minimizing the negative]
        loss = self.criterion(discReal, torch.ones_like(discReal))
        # corresponds to maximising log(1-D(G(z))) [minimizing the negative]
        loss += self.criterion(discFake, torch.zeros_lie(discFake))
        return loss/2, fake
    

    def GenLoss(self, fake):
        """Graph retention not needed here
           max log(D(G(z))) is computationally better in implementation
           (provided in the paper itself) than the proposed loss for 
           Generator in the paper.
        """

        check = self.disc(fake).view(-1)

        # corresponds to maximsing log(D(G(z))) [minizing the negative]
        loss = self.criterion(check, torch.ones_like(check))
        return loss


        

