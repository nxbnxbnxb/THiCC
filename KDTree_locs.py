old/vorovari/mesh___vorovari.py:206:  IDXES=1       # quirk of scipy.spatial.cKDTree class;   0th result is the distances,   1th is indices
old/vorovari/mesh___vorovari.py:216:  KDTree      = scipy.spatial.cKDTree(          vor.points,copy_data=True) # copy_data=True b/c unintended side effects suck.   There may be memory shortages b/c of copy_data=True, though.  
old/vorovari/mesh___vorovari.py:217:  # TODO:  consider use of KDTree again  (high space complexity)
old/vorovari/mesh___vorovari.py:236:    neighbors_idxes = KDTree.query(pt,k=K)[IDXES]
old/vorovari/norms_____by_voro_poles.py:132:      # NOTE: I thought about throwing a KDTree() in here to help me sort through the vorohedron's vertices, but there aren't too many of them
old/rect_prism___voro.py:76:  IDXES=1       # quirk of scipy.spatial.cKDTree class;   0th result is the distances,   1th is indices
old/rect_prism___voro.py:86:  KDTree      = scipy.spatial.cKDTree(          vor.points,copy_data=True) # copy_data=True b/c unintended side effects suck.   There may be memory shortages b/c of copy_data=True, though.  
old/rect_prism___voro.py:87:  # TODO:  consider use of KDTree again  (high space complexity)
old/rect_prism___voro.py:96:      neighbors_idxes = KDTree.query(pt,k=K)[IDXES]
old/mesh.py:205:  IDXES=1       # quirk of scipy.spatial.cKDTree class;   0th result is the distances,   1th is indices
old/mesh.py:215:  KDTree      = scipy.spatial.cKDTree(          vor.points,copy_data=True) # copy_data=True b/c unintended side effects suck.   There may be memory shortages b/c of copy_data=True, though.  
old/mesh.py:216:  # TODO:  consider use of KDTree again  (high space complexity)
old/mesh.py:235:    neighbors_idxes = KDTree.query(pt,k=K)[IDXES]
measure.py:660:        2.  Is it worth the time complexity investment in a  KDTree?
measure.py:709:  # NOTe: kdtree?  octree?
mesh.py:205:  IDXES=1       # quirk of scipy.spatial.cKDTree class;   0th result is the distances,   1th is indices
mesh.py:215:  KDTree      = scipy.spatial.cKDTree(          vor.points,copy_data=True) # copy_data=True b/c unintended side effects suck.   There may be memory shortages b/c of copy_data=True, though.  
mesh.py:216:  # TODO:  consider use of KDTree again  (high space complexity)
mesh.py:235:    neighbors_idxes = KDTree.query(pt,k=K)[IDXES]





































































