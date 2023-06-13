from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from neuro_defs import vk
from vk_api.utils import get_random_id


def create_keyboard(id, text, response="start", users=None):
    try:
        keyboard = VkKeyboard(one_time=True)
        if response == "not_that" or response == "help_me" or response == "rss" or response == "callhuman" or response == "flood":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Ссылка на ВК', "https://vk.com/bramind002")
        elif response == "grifon":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Стикеры с грифоном в ТГ', "https://t.me/addstickers/rtumirea")
        elif response == "psychology" or response == "danger" or response == "feedback_bad" or response == "motivation" or response == "psycho" or response == "psychologist":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Психологическая служба',
                                         "https://student.mirea.ru/psychological_service/staff/")
        elif response == "map" or response == "location":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Навигатор', "https://ischemes.ru/group/rtu-mirea/vern78")
        elif response == "rules":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Устав', "https://www.mirea.ru/upload/medialibrary/d0e/Ustav-Novyy.pdf")
            keyboard.add_openlink_button('Правила внутреннего распорядка', "https://www.mirea.ru/docs/125641/")
            keyboard.add_line()
            keyboard.add_openlink_button('Этический кодекс',
                                         "https://student.mirea.ru/regulatory_documents/file/3f9468db49ffd14fe96c0d28d8c056bf.pdf")
        elif response == "museums":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Подробнее о музеях',
                                         "https://www.mirea.ru/about/history-of-the-university/the-museum-mirea/")
        elif response == "obhodnoy":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Про физкультуру', "https://student.mirea.ru/help/section/physical_education/")
        elif response == "work":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Центр карьеры', "https://career.mirea.ru/")
        elif response == "website":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Сайт МИРЭА', "https://www.mirea.ru/")
        elif response == "military" or response == "для-военкомата":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Памятка военнообязанному', "https://student.mirea.ru/help/section/conscript/")
        elif response == "rectorate" or response == "kudj" or response == "sigov":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Ректорат', "https://www.mirea.ru/about/administration/rektorat/")
        elif response == "library":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Сайт библиотеки', "https://library.mirea.ru/")
        elif response == "office":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('СтудОфис', "https://student.mirea.ru/services/")
        elif response == "scholarship":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Размер стипендии',
                                         "https://student.mirea.ru/scholaship_support/scholarships/state_academic_support/")
        elif response == "social-money":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Материальная помощь', "https://vk.com/topic-42869722_48644800")
            keyboard.add_openlink_button('Бланки', "https://student.mirea.ru/statement/")
        elif response == "subsidy":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Вопросы по дотациям', "https://vk.com/@rtuprofkom-voprosy-po-dotaciyam")
        elif response == "rzhd":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('РЖД-бонус', "https://vk.com/@rtuprofkom-rzhd-bonus-dlya-studentov")
        elif response == "hostel" or response == "hostel-contest":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Подробнее об общежитиях', "https://student.mirea.ru/hostel/campus/")
        elif response == "vuz" or response == "vuc":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('ВУЦ', "https://vuc.mirea.ru/")
        elif response == "expedition":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Экспедиционный корпус', "https://vuc.mirea.ru/ekspeditsionnyy-korpus/")
        elif response == "diving":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Дайвинг клуб', "https://vuc.mirea.ru/kluby/dayving/")
        elif response == "metodichka":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button('Методичка первокурсника',
                                         "https://student.mirea.ru/help/file/metod_perv_2022.pdf")
        elif response == "maps":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Карты кампусов",
                                         "https://student.mirea.ru/help/file/metod_perv_2022.pdf#page=15")
        elif response == "double-diploma":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Программа двойного диплома",
                                         "https://www.mirea.ru/international-activities/training-and-internships/")
        elif response == "car":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Подготовка водителей",
                                         "https://www.mirea.ru/about/the-structure-of-the-university/educational-scientific-structural-unit/driving-school-mstu-mirea/")
        elif response == "other-language":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Ссылка", "https://language.mirea.ru/")
        elif response == "business" or response == "softskill":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Ссылка на группу", "https://vk.com/ntv.mirea")
        elif response == "забыл-вещи":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Бюро находок", "https://vk.com/public79544978")
        elif response == "science":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Научные сообщества", "https://student.mirea.ru/student_scientific_society/")
            keyboard.add_openlink_button("Группа в ВК", "https://vk.com/mirea_smu")
        elif response == "constructor" or response == "accelerator":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Акселератор", "https://project.mirea.ru/")
        elif response == "uvisr":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("УВИСР", "https://student.mirea.ru/")
        elif response == "student-union":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа в ВК", "https://vk.com/sumirea")
        elif response == "media-school":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Медиашкола", "https://vk.com/mediaschool_sumirea")
        elif response == "volunteer":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Волонтёрский центр", "https://vk.com/vcrtumirea")
        elif response == "atmosfera":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Атмосфера", "https://vk.com/atmosfera")
        elif response == "apriori":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Априори", "https://vk.com/apriori.moscow")
        elif response == "counselor":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Атмосфера", "https://vk.com/atmosfera")
            keyboard.add_openlink_button("Априори", "https://vk.com/apriori.moscow")
        elif response == "rescue" or response == "rescue-contacts":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа в ВК", "https://vk.com/csovsks")
        elif response == "vector":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Вектор", "https://vk.com/vector_mirea")
        elif response == "rtuitlab":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("RTUITlab", "https://vk.com/rtuitlab")
        elif response == "group-it":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИИТ в ВК", "https://vk.com/it_sumirea")
        elif response == "group-iii":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИИИ в ВК", "https://vk.com/iii_sumirea")
        elif response == "group-iri":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИРИ в ВК", "https://vk.com/iri_sumirea")
        elif response == "group-ikb":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИКБ в ВК", "https://vk.com/ikb_sumirea")
        elif response == "group-itu":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИТУ в ВК", "https://vk.com/itu_sumirea")
        elif response == "group-itht":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИТХТ в ВК", "https://vk.com/itht_sumirea")
        elif response == "group-iptip":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ИПТИП в ВК", "https://vk.com/iptip__sumirea")
        elif response == "group-kpk":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа КПК в ВК", "https://vk.com/college_sumirea")
        elif response == "rating":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_button("Лайк👍", color=VkKeyboardColor.POSITIVE)
            keyboard.add_button("Дизлайк👎", color=VkKeyboardColor.NEGATIVE)
        elif response == "work":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Центр Карьеры", "https://vk.com/careercenterrtumirea")
        elif response == "graduate-union":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Ассоциация выпускников", "https://student.mirea.ru/graduate/")
        elif response == "radio":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Радиорубка и Радиолаб", "https://vk.com/rtu.radio")
        elif response == "admin":
            keyboard = VkKeyboard(one_time=True)
            keyboard.add_button("1.Вывести все темы", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("2.Вывести всю информацию по теме", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("3.Найти тему по вопросу", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("4.Вывести всю тему по вопросу", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("5.Статистика и рейтинг", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("6.Управление темами", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("7.Переобучить модель", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("8.Выход", color=VkKeyboardColor.NEGATIVE)
        elif response == "edit":
            keyboard = VkKeyboard(one_time=True)
            keyboard.add_button("1.Добавить тему", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("2.Удалить тему", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("3.Добавить ответ к теме", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("4.Добавить вопрос к теме", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("5.Удалить вопрос у темы", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("6.Удалить ответ у темы", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("7.Вернуться", color=VkKeyboardColor.NEGATIVE)
        elif response == "statistic":
            keyboard = VkKeyboard(one_time=True)
            keyboard.add_button("1.Вывести количество тем", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("2.Рейтинг бота", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("3.Количество вопросов", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("4.Вывести количество пользователей", color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button("5.Вернуться", color=VkKeyboardColor.NEGATIVE)
        elif response == "yesno":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_button("Да", color=VkKeyboardColor.POSITIVE)
            keyboard.add_button("Нет", color=VkKeyboardColor.NEGATIVE)
        elif response == "uch-otd" or response == "учебный-отдел-ит":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Учебный отдел",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/contacts/")
        elif response == "профсоюз":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Профсоюз", "https://vk.com/rtuprofkom")
        elif response == "center_culture":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Центр культуры", "https://student.mirea.ru/center_culture/creativity/")
            keyboard.add_openlink_button("Группа в ВК ЦКТ", "https://vk.com/cktmirea")
        elif response == "ври-лес" or response == "школа-леса":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("ВРИ Лес", "https://vk.com/vri_les")
        elif response == "швизис":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("ШВИЗИС", "https://vk.com/shvizis")
        elif response == "заявление-в-студ":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Вступить в СтудСоюз", "https://sumirea.ru/connect/")
        elif response == "отдел-по-работе-общежитие":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Работа с общежитиями", "https://student.mirea.ru/about/section1/")
        elif response == "справочник":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Справочник", "https://tel.mirea.ru/")
        elif response == "кафедра-общей-информатики":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institut-iskusstvennogo-intellekta/the-structure-of-the-institute/chair-of-general-informatics/")
        elif response == "вт" or response == "платонова":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/kvt_mirea")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/department-of-computer-engineering/")
        elif response == "мосит" or response == "головин":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/mireamosit")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/department-of-mathematical-provision-and-standardization-of-information-technology/")
        elif response == "иппо" or response == "болбаков":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/ippo_it")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/department-of-instrumental-and-applied-software/")
        elif response == "ппи" or response == "ппи-зав":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/ppi_it")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/the-department-of-practical-and-applied-computer-science/")
        elif response == "кафедра-пм" or response == "дзержинский":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/kafprimat")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/the-department-of-applied-mathematics/")
        elif response == "кис" or response == "адрианова":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Группа ВК", "https://vk.com/kis_it_mirea")
            keyboard.add_openlink_button("Подробнее",
                                         "https://www.mirea.ru/education/the-institutes-and-faculties/institute-of-information-technology/the-structure-of-the-institute/the-department-of-corporate-information-systems/")
        elif response == "программа-обмена":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Отд.Международного сотрудничества",
                                         "https://www.mirea.ru/about/the-structure-of-the-university/administrative-structural-unit/the-department-of-international-relations/the-department-of-international-cooperation/")
        elif response == "положение-элитной" or response == "отчисление-с-элитной":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Положение Элитной Подготовки",
                                         "https://www.mirea.ru/upload/iblock/555/scb628vl1c1v3ah22653z0grta7pz3fd/pr_1179_10_09_2020_Polozhenie-po-EP.pdf")
        elif response == "стипендия-по-приоритетным-направлениям":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Перечень", "https://base.garant.ru/70842752/#block_3")
        elif response == "страйкбол":
            keyboard = VkKeyboard(inline=True)
            keyboard.add_openlink_button("Страйкбольный клуб", "https://vk.com/rtuairsoftvuc")
        elif users is not None:
            if users[id].language == "en":
                keyboard = VkKeyboard(one_time=False)
                keyboard.add_button('Schedule', color=VkKeyboardColor.PRIMARY)
                keyboard.add_line()
                keyboard.add_button('University map', color=VkKeyboardColor.PRIMARY)
                keyboard.add_line()
                keyboard.add_button('Mailing', color=VkKeyboardColor.PRIMARY)
                keyboard.add_line()
                keyboard.add_button('Retake schedule', color=VkKeyboardColor.PRIMARY)
                keyboard.add_line()
                keyboard.add_button('What do you can?', color=VkKeyboardColor.PRIMARY)
                keyboard.add_line()
                keyboard.add_button('Feedback', color=VkKeyboardColor.PRIMARY)
                keyboard.add_button('Rate the bot', color=VkKeyboardColor.PRIMARY)
                keyboard.add_line()
                keyboard.add_button('Русский', color=VkKeyboardColor.PRIMARY)
        else:
            keyboard = VkKeyboard(one_time=False)
            keyboard.add_button('Расписание', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('Карта Университета', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('Рассылка', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('Расписание пересдач', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('Что ты умеешь?', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('Обратная связь', color=VkKeyboardColor.PRIMARY)
            keyboard.add_button('Оценить бота', color=VkKeyboardColor.PRIMARY)
            keyboard.add_line()
            keyboard.add_button('English', color=VkKeyboardColor.PRIMARY)
        vk.messages.send(
            user_id=id,
            random_id=get_random_id(),
            message=text, keyboard=keyboard.get_keyboard())
    except BaseException as Exception:
        print(Exception)
        return
