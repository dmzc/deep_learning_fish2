# NumPy 学习资料

1\. 《NumPy官方指南（第二版）》

原名：《Guide to NumPy, 2nd Edition》

优点：NumPy创始人亲笔，唯一官方**设计者视角**一手资料。完整覆盖初代ndarray、stride、dtype、广播、ufunc底层架构，收录全套初代C API与迭代器源码实现，支持底层自定义拓展，是NumPy初代源码研读的权威原版蓝本。

缺点：基于2006年初代1\.0版本，无新版架构特性，存在时效性局限。

预期收获：掌握NumPy初代底层架构与C API核心逻辑，读懂原生源码，具备底层自定义拓展能力。

资源链接：https://static\.scipy\.org/doc/\_static/numpybook\.pdf

2\. 《NumPy数组编程（Nature 2020）》

原名：《Array programming with NumPy》（Nature 2020）

优点：NumPy核心开发者顶会综述，**唯一官方十五年迭代总结**。串联新旧版本架构差异，梳理现代优化思路与生态设计理念，补足初代书籍的新版空白。

缺点：偏重宏观架构综述，无底层源码、实操细节。

预期收获：理清NumPy版本迭代脉络，吃透新旧架构差异，建立完整的现代NumPy架构认知。

资源链接：无公开免费完整版

3\. 《Python数据科学手册》

原名：《Python Data Science Handbook》

优点：通俗透彻拆解NumPy核心底层原理，清晰讲解内存布局、广播、向量化高性能核心逻辑，适合快速搭建标准化底层认知。

缺点：仅讲解上层原理，不涉及C底层源码，无法深度源码剖析。

预期收获：夯实NumPy底层理论基础，吃透向量化、内存布局、广播核心原理，规避实操误区。

资源链接：https://jakevdp\.github\.io/PythonDataScienceHandbook/

4\. 《优雅的SciPy》

原名：《Elegant SciPy》

优点：核心开发者撰写，聚焦数组编程范式与顶层设计思想，复盘API优劣与工程取舍，拔高架构认知。

缺点：只讲设计理念，无底层源码拆解内容。

预期收获：掌握数组编程范式与官方工程设计思路，提升代码架构思维，写出高效规范的向量化代码。

5\. 《数值Python科学计算》

原名：《Numerical Python》

优点：独有Cython、ctypes底层交互实操案例，打通Python层与ndarray内存底层衔接，适配工程落地与源码拓展学习。

缺点：侧重工程实操，无系统性C源码剖析内容。

预期收获：掌握Cython、ctypes底层交互方法，具备NumPy源码拓展与底层工程落地实操能力。