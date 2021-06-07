from tspd_mfea_pkg import TSPD_MFEA

if __name__ == '__main__':
    tspd_mfea = TSPD_MFEA(3, 15)
    pop = tspd_mfea.generate_pop()
    for i, idvd in enumerate(pop):
        print("i=", i)
        print(idvd)
