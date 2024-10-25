import pywaves as pw


def main():
    pulsewave_path = "/Users/felixyu/Documents/Remote-Sensing/Neon-Pls-Wvs-Las/NEON_D10_ARIK_DP1_L001-1_2022070915_1.pls"
    pulsewave = pw.openPLS(pulsewave_path)
    wave = pulsewave.get_waves(0)
    wave.plot()


if __name__ == '__main__':
    main()
