import Apps from './Apps'
import { getLocaleOnServer } from '@/i18n/server'
import { useTranslation } from '@/i18n/i18next-serverside-config'

const AppList = async () => {
  const locale = getLocaleOnServer()
  const { t } = await useTranslation(locale, 'app')

  return (
    <div className='flex flex-col overflow-auto bg-gray-100 shrink-0 grow'>
      <Apps />

    </div >
  )
}

export default AppList
