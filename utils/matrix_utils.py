




def dist_matrix(x, y, p=2):
    x_y = x.unsqueeze(1) - y.unsqueeze(0)
    if   p == 1:
        return   x_y.norm(dim=2)
    elif p == 2:
        return ( x_y ** 2).sum(2)
    else:
        return x_y.norm(dim=2)**(p/2)


