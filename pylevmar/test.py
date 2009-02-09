import levmar
from math import sqrt, pow

def func(estimate, measurements, extra):
    errors = []
    #print 'est: %.4f %.4f %.4f %.4f' % estimate
    for i in range(len(measurements)):
        err = sqrt((estimate[0] - extra[i*4])**2 +
                   (estimate[1] - extra[i*4+1])**2 +
                   (estimate[2] - extra[i*4+2])**2)
        err -= 0.299795458 * (extra[i*4+3] - estimate[3])
        errors.append(float(err))
    return errors

def jacf(estimate, measurements, extra):
    jac = []

    for i in range(len(measurements)):
        denom = pow((estimate[0] - extra[i*4])**2 +
                    (estimate[1] - extra[i*4+1])**2 +
                    (estimate[2] - extra[i*4+2])**2, -0.5)

        du_dx1 = (estimate[0] - extra[i*4]) * denom
        du_dx2 = (estimate[1] - extra[i*4+1]) * denom
        du_dx3 = (estimate[2] - extra[i*4+2]) * denom
        du_dx4 = 0.299795458
        jac += [du_dx1, du_dx2, du_dx3, du_dx4]

    return jac
        
def main():
    for line in open('../levmar-2.1.3/sample.data'):
        bits = line.split()
        if len(bits) < 20:
            break

        initial = (0.0, 0.0, 0.0, 0.0)
        measurement = (0.0, 0.0, 0.0, 0.0)
        Ab = map(float, bits[:16])
        #iters, result = levmar.ddif(func, initial, measurement, 5000,
        #                            data = Ab)
        result, iters, stats = levmar.dder(func, jacf, initial, measurement,
                                           5000, data = Ab)
        if result:
            ex, ey, ez, et = result
            tx, ty, tz, tt = map(float, bits[16:20])
            print '%.4f %.4f %.4f (%.4f %.4f %.4f) %d' % \
                  (ex, ey, ez,  tx, ty, tz, iters)

        ##chk = levmar.dchkjac(func, jacf, (0.5, 0.5, 0.5, 10.0), 4, Ab)        
        
if __name__ == "__main__":
    main()


            
        
