import gymnasium as gym  # مكتبة Gymnasium لإنشاء بيئات تعلم التعزيز
import pygame  # مكتبة Pygame لرسم اللعبة والتفاعل مع الرسومات
import random  # مكتبة Random لتوليد أرقام عشوائية (مثل تحديد موقع الكرة بشكل عشوائي)
import numpy as np  # مكتبة Numpy للتعامل مع المصفوفات والعمليات العددية
from stable_baselines3 import DQN  # استيراد خوارزمية DQN للتعلم المعزز من مكتبة stable_baselines3
from stable_baselines3.common.env_checker import check_env  # للتحقق من صحة البيئة التي نعمل عليها
from gymnasium import spaces  # استيراد spaces من Gym لتحديد فضاءات الحركة والمراقبة

# تعريف بيئة جديدة باستخدام Gym
class CatchTheBallEnv(gym.Env):
    # دالة __init__ هي دالة التهيئة التي تعمل عند إنشاء الكائن
    def __init__(self):
        super(CatchTheBallEnv, self).__init__()  # تهيئة الكلاس الرئيسي الذي يرث منه (gym.Env)

        # إعداد خصائص اللعبة (أبعاد الشاشة، السلة، الكرة، السرعات)
        self.width = 600  # عرض النافذة (الشاشة)
        self.height = 400  # ارتفاع النافذة (الشاشة)
        self.basket_width = 100  # عرض السلة
        self.basket_height = 20  # ارتفاع السلة
        self.ball_size = 20  # حجم (قطر) الكرة
        self.basket_speed = 40  # سرعة السلة عند تحركها
        self.ball_speed = 40  # سرعة سقوط الكرة

        # تحديد فضاء الحركة (ثلاثة حركات: يسار، ثبات، يمين)
        self.action_space = spaces.Discrete(3)  # تحديد عدد الحركات الممكنة: اليسار، الثبات، اليمين

        # تحديد فضاء المراقبة (موقع السلة والكرة على المحور الأفقي والرأسي)
        self.observation_space = spaces.Box(low=0.0, high=max(self.width, self.height), shape=(3,), dtype=np.float32)

        # تهيئة Pygame (إعداد نافذة اللعبة)
        self._init_pygame()

        # إعادة تعيين اللعبة إلى الحالة الأولية
        self.reset()

    # إعداد نافذة اللعبة باستخدام مكتبة Pygame
    def _init_pygame(self):
        pygame.init()  # تهيئة مكتبة Pygame
        self.screen = pygame.display.set_mode((self.width, self.height))  # إعداد نافذة اللعبة بأبعاد محددة
        pygame.display.set_caption("Catch The Ball")  # تعيين عنوان نافذة اللعبة

    # إعادة تعيين حالة اللعبة عند بداية أو إعادة تشغيل اللعبة
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # إعادة التعيين العامة من Gym
        self.basket_x = float((self.width - self.basket_width) // 2)  # وضع السلة في منتصف الشاشة
        self.ball_x = float(random.randint(0, self.width - self.ball_size))  # وضع الكرة بشكل عشوائي على المحور الأفقي
        self.ball_y = float(0)  # بدء الكرة من أعلى الشاشة (الموقع y=0)
        self.state = np.array([self.basket_x, self.ball_x, self.ball_y], dtype=np.float32)  # الحالة الأولية للعبة
        return self.state, {}  # إرجاع حالة اللعبة

    # تحديث اللعبة بناءً على الحركة التي يقوم بها اللاعب (أو الذكاء الاصطناعي)
    def step(self, action):
        # تحديث موقع السلة بناءً على الحركة التي يتم اتخاذها
        if action == 0:  # إذا كانت الحركة إلى اليسار
            self.basket_x -= self.basket_speed  # تحرك السلة إلى اليسار
        elif action == 2:  # إذا كانت الحركة إلى اليمين
            self.basket_x += self.basket_speed  # تحرك السلة إلى اليمين

        # التأكد من أن السلة لا تتجاوز حدود الشاشة
        self.basket_x = np.clip(self.basket_x, 0, self.width - self.basket_width)

        # تحديث موقع الكرة بحيث تسقط لأسفل بمقدار سرعة الكرة
        self.ball_y += self.ball_speed

        # إعداد قيم للتحقق إذا انتهت اللعبة ولحساب المكافأة
        done = False  # لم تنتهِ اللعبة بعد
        truncated = False  # اللعبة لم تُقطع
        reward = 0  # المكافأة الابتدائية

        # التحقق إذا وصلت الكرة إلى أسفل الشاشة
        if self.ball_y > self.height - self.basket_height:
            # إذا كانت السلة تحت الكرة، يتم منح مكافأة
            if self.basket_x < self.ball_x < self.basket_x + self.basket_width:
                reward = 1  # المكافأة إذا التقط اللاعب الكرة
            else:
                reward = -1  # العقوبة إذا لم يتم التقاط الكرة
            done = True  # اللعبة تنتهي إذا وصلت الكرة لأسفل

        # إذا لم تنتهِ اللعبة بعد، لكن السلة تحت الكرة
        else:
            if self.basket_x < self.ball_x < self.basket_x + self.basket_width:
                reward = 0.1  # مكافأة صغيرة لتشجيع السلة على البقاء تحت الكرة

        # تحديث حالة اللعبة
        self.state = np.array([self.basket_x, self.ball_x, self.ball_y], dtype=np.float32)

        # إذا انتهت اللعبة، يتم إعادة تعيينها
        if done:
            obs, _ = self.reset()  # إعادة تعيين اللعبة
            return obs, reward, done, truncated, {}  # إرجاع حالة اللعبة بعد إعادة التعيين

        return self.state, reward, done, truncated, {}  # إرجاع حالة اللعبة

    # دالة لرسم اللعبة وتحديث الشاشة باستخدام Pygame
    def render(self, mode="human"):
        if mode == "human":
            # معالجة الأحداث في Pygame مثل إغلاق النافذة
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()  # إغلاق اللعبة إذا تم الضغط على إغلاق
                    return

            # مسح الشاشة بلون أبيض
            self.screen.fill((255, 255, 255))

            # رسم السلة بلون أخضر
            pygame.draw.rect(self.screen, (0, 255, 0), (self.basket_x, self.height - self.basket_height, self.basket_width, self.basket_height))

            # رسم الكرة بلون أحمر
            pygame.draw.circle(self.screen, (255, 0, 0), (int(self.ball_x), int(self.ball_y)), self.ball_size)

            # تحديث الشاشة بعد الرسم
            pygame.display.flip()

            # الانتظار قليلاً بين كل إطار (100 مللي ثانية)
            pygame.time.wait(100)

    # دالة لإغلاق اللعبة
    def close(self):
        pygame.quit()  # إغلاق Pygame وإنهاء اللعبة

# التحقق من صحة البيئة باستخدام دالة check_env من Gym
env = CatchTheBallEnv()
check_env(env)

# تدريب نموذج DQN (شبكة Q العميقة) باستخدام بيئة Catch The Ball
model = DQN("MlpPolicy", env, verbose=1, exploration_fraction=0.2, exploration_final_eps=0.02)

# تدريب النموذج لمدة 20000 خطوة
model.learn(total_timesteps=20000)

# حفظ النموذج بعد التدريب
model.save("catch_the_ball_dqn")

# تحميل النموذج المدرب مسبقًا
model = DQN.load("catch_the_ball_dqn")

# تشغيل اللعبة واختبار النموذج المدرب
obs, _ = env.reset()  # إعادة تعيين اللعبة وبدءها من البداية
for _ in range(1000):  # تكرار العملية 1000 مرة
    for event in pygame.event.get():  # التحقق من أحداث Pygame مثل إغلاق النافذة
        if event.type == pygame.QUIT:
            env.close()  # إغلاق اللعبة إذا تم الضغط على إغلاق
            quit()  # إنهاء البرنامج

    action, _states = model.predict(obs)  # توقع الحركة التالية باستخدام النموذج المدرب
    obs, rewards, done, truncated, _ = env.step(action)  # تنفيذ الحركة وتحديث حالة اللعبة
    env.render()  # رسم اللعبة على الشاشة

    if done:  # إذا انتهت اللعبة، يتم إعادة تعيينها
        obs, _ = env.reset()

# إغلاق اللعبة عند الانتهاء
env.close()
