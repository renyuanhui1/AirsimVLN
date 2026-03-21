1.在运行ue和colosseum插件的时候，发现当ue位于后台时画面比较卡顿，如何设置

在 UE 编辑器界面操作：
点击菜单栏的 Edit (编辑) -> Editor Preferences (编辑器偏好设置)。
在左侧搜索框输入 "Performance" (性能)。
找到 "Use Less CPU when in Background" (处于后台时使用较少 CPU) 这一项。
取消勾选它。

这是解决从 Blender/Google Maps 导入模型最有效的方法。它会让 UE 放弃那个“大箱子”网格，转而根据你模型的实际形状（地面是地面，屋顶是屋顶）来计算碰撞。

在 Content Browser 中双击打开你的地图 FBX（Static Mesh）。

在右侧的 Details 面板中，找到 Static Mesh Settings 部分。

找到 Collision Complexity（碰撞复杂度）选项。

在下拉菜单中选择：Use Complex Collision As Simple（将复杂碰撞用作简单碰撞）。

保存模型。

效果：这样一来，无人机就能停在地面上（因为地面有碰撞），但起飞后上方是空的（因为上方没有网格），不会再撞到“隐形天花板”。