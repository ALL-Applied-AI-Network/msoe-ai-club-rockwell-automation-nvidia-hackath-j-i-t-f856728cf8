import argparse,json,math,os,random,traceback
ap=argparse.ArgumentParser()
for k,v in(("n",6),("scale",2),("subframes",24),("seed",0)):ap.add_argument("--"+k,type=int,default=v)
ap.add_argument("--out",default="/job/out")
a=ap.parse_args()
os.makedirs(a.out+"/images",exist_ok=True)
CJ=json.load(open("/job/in/cameras/CapSealCam.json"))
S=a.scale
RW,RH=CJ["render_px"][0]*S,CJ["render_px"][1]*S
W,H=CJ["final_image_px"][0]*S,CJ["final_image_px"][1]*S
F,K1=CJ["focal_px_final"]*S,CJ["lens_k1"]
from isaacsim import SimulationApp
kit=SimulationApp({"headless":True,"renderer":"PathTracing","width":RW,"height":RH})
try:
 from isaacsim.core.utils.extensions import enable_extension
 enable_extension("omni.replicator.core")
 import numpy as np,cv2
 import omni.usd
 import omni.replicator.core as rep
 from pxr import Gf,Usd,UsdGeom
 ctx=omni.usd.get_context()
 if not ctx.open_stage("/job/in/stations/tray_inspection/tray_inspection.usdc"):raise RuntimeError("open")
 while ctx.get_stage_loading_status()[2]>0:kit.update()
 for _ in range(10):kit.update()
 stage=ctx.get_stage()
 stage.SetEditTarget(stage.GetSessionLayer())
 tc=Usd.TimeCode.Default()
 Rm=np.array(CJ["R_world_to_cv"]);cpos=np.array(CJ["position_m"])
 xl,yl,zl=Rm[0],-Rm[1],-Rm[2]
 M=Gf.Matrix4d(*[float(v) for r in(xl,yl,zl) for v in(*r,0.0)],float(cpos[0]),float(cpos[1]),float(cpos[2]),1.0)
 cam=UsdGeom.Camera.Define(stage,"/GenCam")
 cam.GetHorizontalApertureAttr().Set(36.0)
 cam.GetVerticalApertureAttr().Set(36.0*RH/RW)
 cam.GetFocalLengthAttr().Set(float(CJ["focal_px_final"]/CJ["render_px"][0]*36.0))
 cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.05,20.0))
 UsdGeom.Xformable(cam).AddTransformOp().Set(M)
 rp=rep.create.render_product("/GenCam",(RW,RH))
 ann=rep.AnnotatorRegistry.get_annotator("rgb");ann.attach([rp])
 rep.orchestrator.set_capture_on_play(False)
 kit.update()
 Km=np.array([[F,0,W/2],[0,F,H/2],[0,0,1.0]]);D=np.array([K1,0,0,0,0.0])
 uu,vv=np.meshgrid(np.arange(W,dtype=np.float32),np.arange(H,dtype=np.float32))
 und=cv2.undistortPoints(np.stack([uu.ravel(),vv.ravel()],1)[:,None,:],Km,D).reshape(H,W,2)
 mapx=(und[...,0]*F+RW/2).astype(np.float32)
 mapy=(und[...,1]*F+RH/2).astype(np.float32)
 def project(pw):
  xc=Rm@(np.asarray(pw,dtype=float)-cpos)
  x,y=xc[0]/xc[2],xc[1]/xc[2]
  d=1+K1*(x*x+y*y)
  return float(F*x*d+W/2),float(F*y*d+H/2)
 cache=UsdGeom.BBoxCache(tc,[UsdGeom.Tokens.default_,UsdGeom.Tokens.render])
 names=["G%02d"%i for i in range(1,31)]
 info={}
 for n in names:
  rec={}
  for part in("tube","cap"):
   p=stage.GetPrimAtPath("/TrayInspection/%s_%s"%(n,part))
   if not p.IsValid():raise RuntimeError("missing "+n+part)
   w=UsdGeom.Xformable(p).ComputeLocalToWorldTransform(tc)
   pinv=UsdGeom.Xformable(p.GetParent()).ComputeLocalToWorldTransform(tc).GetInverse()
   b=cache.ComputeWorldBound(p).ComputeAlignedBox()
   rec[part]=dict(prim=p,world=w,pinv=pinv,mn=np.array(b.GetMin()),mx=np.array(b.GetMax()))
  info[n]=rec
 for n in names:
  for part in("tube","cap"):
   r=info[n][part]
   xf=UsdGeom.Xformable(r["prim"])
   xf.ClearXformOpOrder()
   r["op"]=xf.AddTransformOp()
   r["op"].Set(r["world"]*r["pinv"])
 uv={n:project((info[n]["cap"]["mn"]+info[n]["cap"]["mx"])/2) for n in names}
 def T(v):return Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*[float(x) for x in v]))
 def Rot(ax,deg):return Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(*[float(x) for x in ax]),float(deg)))
 def Scl(*s):return Gf.Matrix4d(1.0).SetScale(Gf.Vec3d(*[float(x) for x in s]))
 def about(pv,Mx):return T([-x for x in pv])*Mx*T(pv)
 def vis(p,v):
  im=UsdGeom.Imageable(p)
  im.MakeVisible() if v else im.MakeInvisible()
 def apply(n,cls,rng):
  t,c=info[n]["tube"],info[n]["cap"]
  tw,cw,vt,vc=t["world"],c["world"],True,True
  cx,cy=(c["mn"][0]+c["mx"][0])/2,(c["mn"][1]+c["mx"][1])/2
  rad=(c["mx"][0]-c["mn"][0])/2
  if cls=="missing_cap":vc=False
  elif cls=="no_vial":vt=vc=False
  elif cls=="crooked_cap":
   phi=rng.uniform(0,2*math.pi)
   ax=(math.cos(phi),math.sin(phi),0.0)
   u=(-math.sin(phi),math.cos(phi),0.0)
   sg=rng.choice([-1.0,1.0])
   pv=(cx+sg*rad*u[0],cy+sg*rad*u[1],float(c["mn"][2]))
   cw=cw*about(pv,Rot(ax,-sg*rng.uniform(14,40)))
  elif cls=="damaged_vial":
   pv=((t["mn"][0]+t["mx"][0])/2,(t["mn"][1]+t["mx"][1])/2,float(t["mn"][2]))
   th=float(t["mx"][2]-t["mn"][2])
   phi=rng.uniform(0,180)
   sx,sy,sz=rng.uniform(0.6,1.0),rng.uniform(0.6,1.0),rng.uniform(0.7,0.95)
   sq=Rot((0,0,1),-phi)*Scl(sx,sy,sz)*Rot((0,0,1),phi)
   ph=rng.uniform(0,2*math.pi)
   tilt=Rot((math.cos(ph),math.sin(ph),0.0),rng.uniform(8,25))
   tw=tw*about(pv,sq)*about(pv,tilt)
   cw=cw*T((0.0,0.0,-(1-sz)*th))*about(pv,tilt)
  t["op"].Set(tw*t["pinv"]);c["op"].Set(cw*c["pinv"])
  vis(t["prim"],vt);vis(c["prim"],vc)
 LK=stage.GetPrimAtPath("/TrayInspection/key_overhead/key_overhead")
 LF=stage.GetPrimAtPath("/TrayInspection/fill_side/fill_side")
 bk=float(LK.GetAttribute("inputs:intensity").Get())
 bf=float(LF.GetAttribute("inputs:intensity").Get())
 classes=["pass","missing_cap","crooked_cap","damaged_vial","no_vial"]
 weights=[0.45,0.15,0.15,0.10,0.15]
 rng=random.Random(a.seed)
 recs=[]
 for i in range(a.n):
  name="img_%04d"%i
  labels={}
  for n in names:
   cls=rng.choices(classes,weights)[0]
   labels[n]=cls
   apply(n,cls,rng)
  ki,fi=rng.uniform(0.6,1.5),rng.uniform(0.5,2.0)
  col=Gf.Vec3f(1.0,rng.uniform(0.82,1.0),rng.uniform(0.62,1.0))
  LK.GetAttribute("inputs:intensity").Set(bk*ki);LK.GetAttribute("inputs:color").Set(col)
  LF.GetAttribute("inputs:intensity").Set(bf*fi);LF.GetAttribute("inputs:color").Set(col)
  for _ in range(3):kit.update()
  for _ in range(2):rep.orchestrator.step(rt_subframes=a.subframes,delta_time=0.0)
  data=np.asarray(ann.get_data())
  if data.ndim!=3 or data.shape[:2]!=(RH,RW) or data.shape[2]<3:raise RuntimeError("bad shape %s"%(data.shape,))
  img=np.ascontiguousarray(data[...,:3][...,::-1])
  cv2.imwrite("%s/images/%s.png"%(a.out,name),cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT))
  recs.append(dict(image=name,labels=labels,light=dict(key=ki,fill=fi,color=[float(x) for x in col])))
  json.dump(dict(size=[W,H],positions=uv,images=recs),open(a.out+"/labels.json","w"),indent=1)
  print("done",name,flush=True)
except BaseException:
 traceback.print_exc()
 kit.close(exit_code=1)
 raise
kit.close()
