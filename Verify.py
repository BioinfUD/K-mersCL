data = "0 0 0 3 3 3 3 1 0 0 0 2 3 2 1 1"

def to__one_number(data, base=4):
    datos = [int(i) for i in data.split()]
    n = len(datos)
    total = 0
    print datos
    for i in range(n):
      #print i
      exp = int(base**(n-i-1))
      print "Dato  %s, posicion %s, exponente %s, sumador: %s" % (datos[i], i, exp, datos[i]*exp)
      total += datos[i]*exp

    print "Total :%s" % total
    return total

def to_bases(data):
    datos = [int(i) for i in data.split(" ")]
    n = len(datos)
    total = 0
    print datos
    for i in range(n):
      #print i
      exp = int(4**(n-i-1))
      print "Dato  %s, posicion %s, exponente %s, sumador: %s" % (datos[i], i, exp, datos[i]*exp)
      total += datos[i]*exp

    print "Total :%s" % total
    return total
