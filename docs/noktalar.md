# Nokta ve iskelet çizimi

Kutular bir aile, eklemler diğeri. cvflair poz tahmini **yapmaz** — noktalar
senin çalıştırdığın modelden gelir (MediaPipe, YOLO-pose, OpenPose); burada olan
şey taşıyıcı ve indeksler arası bağlantı.

```python
from cvflair import HAND_21, Camera, KeyPoints

cam = Camera(source=0, theme="neon")
for frame in cam.stream():
    points = KeyPoints(xy=el_noktalarini_bul(frame))   # senin modelin
    cam.show(frame, keypoints=points, skeleton=HAND_21)
```

Kutu ve iskelet aynı karede birlikte de çizilebilir:

```python
cam.show(frame, detections, keypoints=points, skeleton="pose")
```

## `KeyPoints`

| Alan | Zorunlu | Ne |
|---|---|---|
| `xy` | evet | `(N, K, 2)`: N iskelet, K nokta, piksel koordinatı. Tek iskelet için `(K, 2)` de olur |
| `confidence` | hayır | `(N, K)` nokta başına güven; eşiğin altındakiler çizilmez |
| `class_id` | hayır | iskelet başına sınıf; renk dağıtımı buna bakar |

```python
KeyPoints.empty(21)                                  # boş
KeyPoints.from_normalized(xy, width=640, height=480) # 0-1 aralığını piksele çevirir
```

`from_normalized` MediaPipe gibi noktaları kareye göre normalize veren
kütüphaneler için: ölçekleme burada yapılır, model bilgisi pakete girmez.

## İskeletler

Bir iskelet düz veridir: bağlanacak nokta indekslerinin listesi.

| Sabit | Nokta | Sıralama |
|---|---|---|
| `HAND_21` | 21 | MediaPipe Hands — bilek 0, başparmak 1-4, işaret 5-8, orta 9-12, yüzük 13-16, serçe 17-20 |
| `POSE_17` | 17 | COCO / YOLO-pose — burun 0, gözler 1-2, kulaklar 3-4, omuzlar 5-6, dirsekler 7-8, bilekler 9-10, kalçalar 11-12, dizler 13-14, ayak bilekleri 15-16 |

Ad olarak da verilebilir: `skeleton="hand"`, `skeleton="pose"`
(`cvflair.SKELETONS` sözlüğünde).

Farklı bir düzenin varsa kendi listeni geçebilirsin:

```python
KANAT = ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5))
cam.show(frame, keypoints=points, skeleton=KANAT)
```

İskelet veriden fazla nokta tarif ediyorsa fazlalık kenarlar atlanır; modelin
verdiği kadarı çizilir.

## Tema ayarları

| Alan | Ne |
|---|---|
| `pose_thickness` | kemik kalınlığı (varsayılan 2) |
| `pose_radius` | eklem noktası yarıçapı (varsayılan 3) |
| `pose_confidence` | bir noktanın çizilmesi için gereken en düşük güven (varsayılan 0,3) |

Kemikler paletin rengini alır. Temada vurgu rengi varsa **eklemler o renge geçer**
— `cyberpunk` bu yüzden renkli kemik + beyaz eklem görünür. `glow` açıkken
kemiklerin arkasına koyulaştırılmış kalın bir geçiş çizilir.

Birden çok iskelet varsa her biri paletten sırayla renk alır; `class_id`
verilirse renk ona göre dağıtılır (iki el aynı renk, kişi başka renk gibi).

## Bozuk noktalar

Kutularda olduğu gibi: `NaN`, sonsuz veya güven eşiğinin altındaki noktalar
çizilmez. O noktaya bağlı kemikler de atlanır, iskeletin geri kalanı çizilmeye
devam eder — model bir eklemi kaybettiğinde çizim durmaz.

## Tarayıcıda denemek

[Tema playground](https://kbycode.github.io/cvflair/) önizlemesinde **iskelet**
kipi var: el ve poz iskeleti canlı çiziliyor, kemik kalınlığı ile eklem boyu
oradan ayarlanıyor. Kutu ayarları o kipte gizleniyor, üretilen kod da iskelet
kullanımını gösteriyor.

## Örnek

[`examples/hand_skeleton.py`](https://github.com/kbycode/cvflair/blob/main/examples/hand_skeleton.py)
kamera görüntüsünün üzerine açılıp kapanan 21 noktalı bir el çizer. El sentetiktir
(ek kurulum gerekmesin diye); gerçek bir elde tek fark noktaların nereden geldiği,
örneğin MediaPipe bağlantısı dosyanın başında yazılı.
