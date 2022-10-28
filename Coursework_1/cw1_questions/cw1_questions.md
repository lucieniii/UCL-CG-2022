## 6 Reality check

###Can be rendered

|                |                      Black Plastic Ball                      |                         Chrome Ball                          |                          Jade Ball                           |
| :------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| **In reality** | <img src="Black Plastic Ball real.png" alt="Black Plastic Ball real" style="zoom:50%;" /> | <img src="Chrome Ball real.png" alt="Chrome Ball real" style="zoom:50%;" /> | <img src="Jade Ball real.png" alt="Jade Ball real" style="zoom:50%;" /> |
|  **Rendered**  | <img src="Black Plastic Ball rendered.png" alt="Black Plastic Ball rendered" style="zoom:50%;" /> | <img src="Chrome Ball rendered.png" alt="Chrome Ball rendered" style="zoom:50%;" /> | <img src="/Users/lucien/Github/UCL-CG-2022/Coursework_1/cw1_questions/Jade Ball rendered.png" alt="Jade Ball rendered" style="zoom:50%;" /> |
|     $k_d$      |                       $0.01,0.01,0.01$                       |                        $0.4,0.4,0.4$                         |                       $0.54,0.89,0.63$                       |
|     $k_s$      |                       $0.50,0.50,0.50$                       |                       $0.77,0.77,0.77$                       |                       $0.32,0.32,0.32$                       |
|      $n$       |                            $32.0$                            |                            $16.8$                            |                            $12.8$                            |

###Cannot be realized

|                        Color Bleeding                        |                    Subsurface Scattering                     |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="cornell_lightup.jpeg" alt="cornell_lightup" style="zoom:50%;" /> | <img src="Skin_Subsurface_Scattering.jpeg" alt="Skin_Subsurface_Scattering" style="zoom:67%;" /> |

The effect of **color bleeding** cannot be rendered because our model doesn't consider the color of light reflected/refracted from one material to another.

Our model also cannot handle **subsurface scattering** which could be seen on some translucent materials, which light would reflect in the materials for sometimes before leaving the material but our model assume that light would leave the surface immediately when reflecting.