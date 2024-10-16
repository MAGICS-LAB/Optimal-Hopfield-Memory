import math
import matplotlib.pyplot as plt

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica"
# })


angles = [109.4712206, 90.0000000, 90.0000000, 77.8695421,  74.8584922, 70.5287794, 66.1468220, 63.4349488, 63.4349488, 57.1367031, 55.6705700, 53.6578501, 52.2443957,
         51.0903285 ,49.5566548,47.6919141, 47.4310362, 45.6132231, 44.7401612, 43.7099642, 43.6907671, 41.6344612, 41.0376616, 40.6776007, 39.3551436, 38.7136512,  38.5971159,  37.7098291, 37.4752140, 36.2545530, 35.8077844, 35.3198076, 35.1897322, 34.4224080, 34.2506607, 33.4890466, 33.1583563,
            32.7290944, 32.5063863, 32.0906244, 31.9834230, 31.3230814, 30.9591635, 30.7818159, 30.7627855, 29.9235851, 29.7529564, 28.2627914, 27.1928300,  26.0698299, 25.1709200,  24.3017225, 23.5530672, 22.7791621, 22.1540232, 21.5945501,21.0312020, 20.5388524,20.1113276, 19.6239931, 19.3240201, 18.8448151, 18.5103522]
points = [i for i in range(4, 51)] + [55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130]


def upper_bound(M, d=3):
    inside= (  (2*math.sqrt(math.pi)/M) * math.gamma( (d+1)/2 )/math.gamma( (d/2) ))
    inside = inside**(1/(d-1))
    return 2*inside

def lower_bound(M, d=3):

    inside = math.sqrt(math.pi)/M
    inside2 = (math.gamma(  (d+1)/2 ))/( math.gamma( (d/2)+1  ) )
    inside = inside * inside2
    inside = inside**(2/(d-1))
    return 0.5*inside

upper = [upper_bound(p) for p in points]
lower = [lower_bound(p) for p in points]

sep = [a*math.pi/180 for a in angles]
sep = [1 - math.cos(s) for s in sep]

font = {'size': 12}

plt.rc('font', **font)


plt.plot(points, sep, label='Ground Truth')
plt.plot(points, upper, label='upper bound')

plt.plot(points, lower, label='lower bound')
plt.xlabel("M")
plt.ylabel("$\Delta_{min}^\Phi$")
plt.title("Separation Bound (d=3)")
plt.legend()
plt.show()
plt.savefig("separation_bound.png", dpi=600)