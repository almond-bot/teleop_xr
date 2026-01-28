import { World, PanelUI, Interactable, DistanceGrabbable, MovementMode } from "@iwsdk/core";
import { Mesh, PlaneGeometry, MeshBasicMaterial, VideoTexture, DoubleSide } from "three";

export class DraggablePanel {
  public entity: any;

  constructor(protected world: World, configPath: string, options: any = {}) {
    this.entity = world.createTransformEntity()
      .addComponent(PanelUI, {
        config: configPath,
        ...options
      })
      .addComponent(Interactable)
      .addComponent(DistanceGrabbable, {
        movementMode: MovementMode.MoveFromTarget
      });
  }

  setPosition(x: number, y: number, z: number) {
    this.entity.object3D.position.set(x, y, z);
  }
}

export class CameraPanel extends DraggablePanel {
  private videoMesh: Mesh | null = null;
  private videoElement: HTMLVideoElement | null = null;

  constructor(world: World) {
    super(world, "./ui/camera.uikitml", {
      maxHeight: 0.6,
      maxWidth: 0.8,
    });
  }

  setVideoTrack(track: MediaStreamTrack) {
    if (this.videoMesh) return; // Already set

    const stream = new MediaStream([track]);
    this.videoElement = document.createElement("video");
    this.videoElement.srcObject = stream;
    this.videoElement.playsInline = true;
    this.videoElement.muted = true; // Required for autoplay
    this.videoElement.style.display = "none";
    document.body.appendChild(this.videoElement);

    this.videoElement.play().catch(e => {
        console.error(`Video play error: ${e}`);
    });

    const texture = new VideoTexture(this.videoElement);
    // Aspect ratio 1.5 roughly
    const geometry = new PlaneGeometry(0.6, 0.4); 
    const material = new MeshBasicMaterial({ map: texture, side: DoubleSide });
    this.videoMesh = new Mesh(geometry, material);
    
    // Position it slightly in front of the panel to avoid z-fighting
    this.videoMesh.position.z = 0.02; 
    // Adjust y to be centered or below title
    this.videoMesh.position.y = -0.1;

    this.entity.object3D.add(this.videoMesh);
  }
}
