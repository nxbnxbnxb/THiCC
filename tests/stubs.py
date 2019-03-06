Wed Mar  6 14:02:26 EST 2019
 measure.py: testing error of HMR mesh: "obj_error()"
  '''
  # Note: Testing whether model is oriented correctly:     It seems it is.
  cube=np.array([ [      0,      0,      0,],
                  [      0,      0, z_max ,],
                  [      0, y_max ,      0,],
                  [      0, y_max , z_max ,],
                  [ x_max ,      0,      0,],
                  [ x_max ,      0, z_max ,],
                  [ x_max , y_max ,      0,],
                  [ x_max , y_max , z_max ,],]).astype("float64")
  p0=vt.data[vt.query(cube[0])[1]]
  p1=vt.data[vt.query(cube[1])[1]]
  p2=vt.data[vt.query(cube[2])[1]]
  p3=vt.data[vt.query(cube[3])[1]]
  p4=vt.data[vt.query(cube[4])[1]]
  p5=vt.data[vt.query(cube[5])[1]]
  p6=vt.data[vt.query(cube[6])[1]]
  p7=vt.data[vt.query(cube[7])[1]]
  pr("p0:",p0)
  pr("p1:",p1)
  pr("p2:",p2)
  pr("p3:",p3)
  pr("p4:",p4)
  pr("p5:",p5)
  pr("p6:",p6)
  pr("p7:",p7)
  '''
