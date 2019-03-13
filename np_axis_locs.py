# TODO: figure out for which of these np functions the "axis=" parameter is necessary.  One of them requires a relatively new version of numpy which we oughta make sure is compatible with all the other **!t (packages) we have installed... was it 1.16?
'''
# NOTE: was it  np.unique?   Had to be something fairly complex/obscure...
                np.count_nonzero()
                np.mean()
                np.sum()?
                np.concatenate()?
                np.max()?
                np.min()?
                np.argmax()?
                np.argmin()?
'''






































































model.py:84:def mask(model, mask, axis='x'):
model.py:116:#====================================  end func def of mask(model, mask, axis='x'):  ====================================================
model.py:127:    show_cross_sections(model, axis='y', freq=250)
Binary file .seg.py.swp matches
viz.py:35:def show_cross_sections(model_3d, axis='z', freq=2):
viz.py:56:      print ("Usage: please input axis x, y, or z in format:\n\n  show_cross_sections([model_name], axis='z')")
viz.py:61:  show_cross_sections(model_3d, axis='x', freq=freq); print ('\n'*3); print ("y: \n\n")
viz.py:62:  show_cross_sections(model_3d, axis='y', freq=freq); print ('\n'*3); print ("z: \n\n")
viz.py:63:  show_cross_sections(model_3d, axis='z', freq=freq); return
old/old_code.py:260:  segmap=np.concatenate((segmap,segmap,segmap),axis=2)
old/old_code.py:410:    show_cross_sections(model, axis='y')
old/old_code.py:431:  counts=np.count_nonzero(mask, axis=1)
old/vorovari/mesh___vorovari.py:95:  inner_pt = np.mean(vertices[:2],axis=0).reshape((1,3))
old/vorovari/mesh___vorovari.py:100:    tetra=np.concatenate((inner_pt,triangle),axis=0)
old/vorovari/mesh___vorovari.py:101:    CoM_tetra=np.mean(tetra,axis=0)
old/vorovari/mesh___vorovari.py:252:        unions_CoM  = np.sum(voro_vols[union_so_far].reshape((neighbor_idx+1,1))  * voro_CoMs[union_so_far],  axis=DOWN) / unions_vol
old/vorovari/mesh___vorovari.py:258:        unions_covar= np.sum( voro_covars[union_so_far] - shifts,  axis=0) # calculation from Alliez paper (covars of unions of vorohedra)
old/vorovari/mesh___vorovari.py:293:        inner_pt  = np.mean(vertices[:FIRST_2_PTS],axis=UNDER).reshape((1,3)) # new tetrahedron vertex.   In the beginning this is good enough because we're guaranteed Vorohedrons are convex
old/vorovari/mesh___vorovari.py:300:          tetra     = np.concatenate((inner_pt,triangle),axis=UNDER)
old/vorovari/mesh___vorovari.py:301:          CoM_tetra = np.mean(tetra,axis=DOWN)
old/vorovari/mesh___vorovari.py:309:          tetra=np.concatenate((CoM_voro.reshape((1,3)),triangle),axis=UNDER)
old/vorovari/mesh___vorovari.py:313:        unions_CoM  = np.sum(voro_vols[union_so_far].reshape((neighbor_idx+1,1))  *voro_CoMs[union_so_far],axis=DOWN) / unions_vol
old/vorovari/mesh___vorovari.py:321:        unions_covar= np.sum(  voro_covars[union_so_far] - shifts,axis=0) # calculation from Alliez paper (covars of unions of vorohedra)
old/vorovari/mesh___vorovari.py:322:        # TODO:                                       is axis=0 right or axis=2?  In the earlier line I wrote 2 and I was probably more awake when I wrote that
old/old_measure.py:64:  distances   = np.sum(np.sqrt(np.abs(locs-bot_left )),axis=1)
old/old_measure.py:66:  distances   = np.sum(np.sqrt(np.abs(locs-bot_right)),axis=1)
old/old_measure.py:155:  return np.sum(np.sqrt(np.sum(np.square(crotch-toes),axis=1)))/TWO_LEGS
old/old_measure.py:168:  toes  =np.concatenate((toes['left_toe'],toes['right_toe']),axis=0)
old/rect_prism___voro.py:109:          unions_CoM  = np.sum(voro_vols[union_so_far].reshape((neighbor_idx+1,1))  * voro_CoMs[union_so_far],  axis=DOWN) / unions_vol
old/rect_prism___voro.py:115:          unions_covar= np.sum( voro_covars[union_so_far] - shifts,  axis=0) # calculation from Alliez paper (covars of unions of vorohedra)
old/rect_prism___voro.py:150:          inner_pt  = np.mean(vertices[:FIRST_2_PTS],axis=UNDER).reshape((1,3)) # new tetrahedron vertex.   In the beginning this is good enough because we're guaranteed Vorohedrons are convex
old/rect_prism___voro.py:157:            tetra     = np.concatenate((inner_pt,triangle),axis=UNDER)
old/rect_prism___voro.py:158:            CoM_tetra = np.mean(tetra,axis=DOWN)
old/rect_prism___voro.py:166:            tetra=np.concatenate((CoM_voro.reshape((1,3)),triangle),axis=UNDER)
old/rect_prism___voro.py:170:          unions_CoM  = np.sum(voro_vols[union_so_far].reshape((neighbor_idx+1,1))  *voro_CoMs[union_so_far],axis=DOWN) / unions_vol
old/rect_prism___voro.py:178:          unions_covar= np.sum(  voro_covars[union_so_far] - shifts,axis=0) # calculation from Alliez paper (covars of unions of vorohedra)
old/rect_prism___voro.py:179:          # TODO:                                       is axis=0 right or axis=2?  In the earlier line I wrote 2 and I was probably more awake when I wrote that
old/test___norm_generation___in_mesh_py.py:11:def plane(slope=1.0, axis='x', n_pts=1000):
old/test___norm_generation___in_mesh_py.py:14:    Independent of changes in the "axis" direction (in other words, if axis='x', the slope is in the yz plane, so at x=0 you would see the same inclined line as you see at x=1).  But I will add some noise as well to properly test the "Voronoi-PCA" algorithm described in Alliez et al. 2007
old/test___norm_generation___in_mesh_py.py:19:  if axis=='x':
old/test___norm_generation___in_mesh_py.py:30:  # TODO: if axis=='y':    , if axis=='z':,  slope
old/test___norm_generation___in_mesh_py.py:34:def noisy_plane(slope=1.0, axis='x', n_pts=1000):
old/test___norm_generation___in_mesh_py.py:37:    Independent of changes in the "axis" direction (in other words, if axis='x', the slope is in the yz plane, so at x=0 you would see the same inclined line as you see at x=1).  But I will add some noise as well to properly test the "Voronoi-PCA" algorithm described in Alliez et al. 2007
old/test___norm_generation___in_mesh_py.py:42:  if axis=='x':
old/test___norm_generation___in_mesh_py.py:53:  # TODO: if axis=='y':    , if axis=='z':,  slope
old/test_mesh.py:11:def plane(slope=1.0, axis='x', n_pts=1000):
old/test_mesh.py:14:    Independent of changes in the "axis" direction (in other words, if axis='x', the slope is in the yz plane, so at x=0 you would see the same inclined line as you see at x=1).  But I will add some noise as well to properly test the "Voronoi-PCA" algorithm described in Alliez et al. 2007
old/test_mesh.py:19:  if axis=='x':
old/test_mesh.py:30:  # TODO: if axis=='y':    , if axis=='z':,  slope
old/test_mesh.py:34:def noisy_plane(slope=1.0, axis='x', n_pts=1000):
old/test_mesh.py:37:    Independent of changes in the "axis" direction (in other words, if axis='x', the slope is in the yz plane, so at x=0 you would see the same inclined line as you see at x=1).  But I will add some noise as well to properly test the "Voronoi-PCA" algorithm described in Alliez et al. 2007
old/test_mesh.py:42:  if axis=='x':
old/test_mesh.py:53:  # TODO: if axis=='y':    , if axis=='z':,  slope
old/mesh.py:94:  inner_pt = np.mean(vertices[:2],axis=0).reshape((1,3))
old/mesh.py:99:    tetra=np.concatenate((inner_pt,triangle),axis=0)
old/mesh.py:100:    CoM_tetra=np.mean(tetra,axis=0)
old/mesh.py:251:        unions_CoM  = np.sum(voro_vols[union_so_far].reshape((neighbor_idx+1,1))  * voro_CoMs[union_so_far],  axis=DOWN) / unions_vol
old/mesh.py:257:        unions_covar= np.sum( voro_covars[union_so_far] - shifts,  axis=0) # calculation from Alliez paper (covars of unions of vorohedra)
old/mesh.py:292:        inner_pt  = np.mean(vertices[:FIRST_2_PTS],axis=UNDER).reshape((1,3)) # new tetrahedron vertex.   In the beginning this is good enough because we're guaranteed Vorohedrons are convex
old/mesh.py:299:          tetra     = np.concatenate((inner_pt,triangle),axis=UNDER)
old/mesh.py:300:          CoM_tetra = np.mean(tetra,axis=DOWN)
old/mesh.py:308:          tetra=np.concatenate((CoM_voro.reshape((1,3)),triangle),axis=UNDER)
old/mesh.py:312:        unions_CoM  = np.sum(voro_vols[union_so_far].reshape((neighbor_idx+1,1))  *voro_CoMs[union_so_far],axis=DOWN) / unions_vol
old/mesh.py:320:        unions_covar= np.sum(  voro_covars[union_so_far] - shifts,axis=0) # calculation from Alliez paper (covars of unions of vorohedra)
old/mesh.py:321:        # TODO:                                       is axis=0 right or axis=2?  In the earlier line I wrote 2 and I was probably more awake when I wrote that
Binary file __pycache__/viz.cpython-36.pyc matches
Binary file .git/objects/pack/pack-1c597920dfb4e79cfe20ef285ddbe49a35c454f9.pack matches
seg.py:234:  segmap  = np.rot90(       np.concatenate((segmap,segmap,segmap),axis=2)        )
tests/test___rotate___on_locs.py:66:  return np.unique(full_locs, axis=0).astype('uint64') # shape == (large_num, 3)
tests/install_tests/test_skimage.py:14:                               ellip_base[2:, ...]), axis=0)
tests/install_tests/test_numpy.py:5:print(np.unique(nine,axis=0))
ref/measure_manually/measure.py:14:#shape=np.concatenate(trapezoid,np.mean(trapezoid,axis=0),axis='x')
ref/face/human_face.py:36:    axis=0)
ref/face/human_face.py:46:  arm_length  = np.max(np.count_nonzero(mask,axis=1))
ref/face/human_face.py:47:  y_arms      = np.argmax(np.count_nonzero(mask,axis=1))
ref/face/human_face.py:48:  # TODO: vectorize this for-loop (into numpy).  Maybe count_nonzero outside the for_loop?  If I remember correctly, np.count_nonzero(...,axis=0) was vectorized recently
rot8.py:24:    return np.concatenate(coords,axis=1).astype('int64') # locations have to be integer (indices)
rot8.py:29:def rot8(model,angle,axis='z'):
rot8.py:151:    return np.unique(full_locs, axis=three_tuples).astype('uint64') # shape == (large_num, 3)      # TODO:  remove the (0,0,0) coords in here
rot8.py:168:#========= end func def of rot8(model,angle,axis='z'): =========
Binary file viz.pyc matches
